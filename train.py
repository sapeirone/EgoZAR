import wandb
import os.path as osp

import torch
import argparse
import numpy as np

from torchmetrics.classification import MulticlassAccuracy
from torchmetrics import MeanMetric
from torch.utils.data import DataLoader

from tqdm import tqdm

from dataset import EK100Dataset
from meters.multitask_accuracy import MultitaskAccuracy

from models.model import EgoZAR

import torch.nn.functional as F

from sklearn.cluster import KMeans

import warnings

import coloredlogs, logging

logging.basicConfig()
coloredlogs.install(level='DEBUG')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

warnings.filterwarnings("ignore")


device = "cuda"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


@torch.no_grad()
def validate(model, loader):

    # top-1 accuracy meters
    verb_top1 = MulticlassAccuracy(num_classes=97, top_k=1, average="micro").to(device)
    noun_top1 = MulticlassAccuracy(num_classes=300, top_k=1, average="micro").to(device)
    actions_top1 = MultitaskAccuracy(num_labels=2, top_k=1).to(device)

    # top-5 accuracy meters
    verb_top5 = MulticlassAccuracy(num_classes=97, top_k=5, average="micro").to(device)
    noun_top5 = MulticlassAccuracy(num_classes=300, top_k=5, average="micro").to(device)
    actions_top5 = MultitaskAccuracy(num_labels=2, top_k=5).to(device)

    for m in model.values():
        m.eval()

    for env_features, features, verb_labels, noun_labels, *_ in tqdm(loader):
        env_features = env_features.to(device)
        features = {k: f.to(device) for k, f in features.items()}
        verb_labels = verb_labels.to(device)
        noun_labels = noun_labels.to(device)
        
        verb_logits, noun_logits = dict(), dict()

        for mod, m in model.items():
            verb_logits[mod], noun_logits[mod], *_ = model[mod](features[mod], env_features)

        verb_logits = torch.stack(list(verb_logits.values())).sum(dim=0)
        noun_logits = torch.stack(list(noun_logits.values())).sum(dim=0)

        # update top-1 accuracy meters
        verb_top1.update(verb_logits, verb_labels)
        noun_top1.update(noun_logits, noun_labels)
        actions_top1.update((verb_logits, noun_logits), (verb_labels, noun_labels))

        # update top-5 accuracy meters
        verb_top5.update(verb_logits, verb_labels)
        noun_top5.update(noun_logits, noun_labels)
        actions_top5.update((verb_logits, noun_logits), (verb_labels, noun_labels))

    return (
        100 * verb_top1.compute(),
        100 * noun_top1.compute(),
        100 * actions_top1.compute(),
        100 * verb_top5.compute(),
        100 * noun_top5.compute(),
        100 * actions_top5.compute(),
    )


def train(
    model,
    loader,
    optimizer,
    scheduler,
    env_labels: torch.Tensor,
    double_ce=False,
    disent_loss_weight=0.0,
    disent_all_modalities=False,
):
    # Classification and disentanglement loss meters
    loss_meter = MeanMetric(nan_strategy="error").to(device)
    disent_loss_meter = MeanMetric(nan_strategy="error").to(device)

    for m in model.values():
        m.train()

    pbar = tqdm(loader, leave=False)
    pbar.set_description("Training...")
    
    for env_features, features, verb_labels, noun_labels, idxs in pbar:
        env_features = env_features.to(device)
        features = {k: f.to(device) for k, f in features.items()}
        verb_labels = verb_labels.to(device)
        noun_labels = noun_labels.to(device)

        # forward pass on each modality separately
        verb_logits, noun_logits = dict(), dict()
        feat_motion, feat_acz = dict(), dict()
        motion_cls, acz_cls = dict(), dict()
        batch_env_labels = env_labels[idxs]
        
        for mod, m in model.items():
            (
                verb_logits[mod],
                noun_logits[mod],
                feat_motion[mod],
                feat_acz[mod],
                motion_cls[mod],
                acz_cls[mod],
            ) = model[mod](features[mod], env_features, batch_env_labels)

        # Compute the classification loss for all the input modalities separately (double CE)
        classification_loss = 0.0
        if double_ce:
            classification_loss += 0.33 * sum(
                0.5 * (F.cross_entropy(vl, verb_labels) + F.cross_entropy(nl, noun_labels))
                for vl, nl in zip(verb_logits.values(), noun_logits.values())
            )

        verb_logits = torch.stack(list(verb_logits.values())).sum(dim=0)
        noun_logits = torch.stack(list(noun_logits.values())).sum(dim=0)

        # Compute the classification loss for the fused logits
        classification_loss += 0.5 * (F.cross_entropy(verb_logits, verb_labels) + F.cross_entropy(noun_logits, noun_labels))
        loss_meter.update(float(classification_loss))

        # Compute the disentanglement loss
        disent_loss = 0.0
        if disent_all_modalities:
            disent_loss = sum(F.cross_entropy(motion_cls[m][batch_env_labels >= 0], batch_env_labels[batch_env_labels >= 0]) for m in model.keys())
            disent_loss = disent_loss / len(model.keys())
        elif "RGB" in model.keys():
            disent_loss = F.cross_entropy(motion_cls["RGB"][batch_env_labels >= 0], batch_env_labels[batch_env_labels >= 0])
        disent_loss_meter.update(float(disent_loss))

        loss = classification_loss

        if disent_loss_weight > 0.0 and "RGB" in model.keys():
            loss += disent_loss_weight * disent_loss
            
        pbar.set_description(f"Training (loss = {loss:.4f})...")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()

    return loss_meter.compute(), disent_loss_meter.compute()


def build_clustering_model(train_dataset, n_clusters, device="cuda"):
    env_features = torch.stack([ef for ef, *_ in train_dataset], dim=0).mean(1).numpy()
    model = KMeans(n_clusters=n_clusters, n_init="auto")
    return torch.from_numpy(model.fit_predict(env_features)).long().to(device)


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    logger.info("Initializing the datasets...")
    train_dataset = EK100Dataset.build_dataset("source", "train", modalities=args.modality, num_segments=args.num_segments, clip_features_prefix=args.clip_features_prefix,)
    val_dataset = EK100Dataset.build_dataset("target", "val", modalities=args.modality, num_segments=args.num_segments, clip_features_prefix=args.clip_features_prefix,)
    train_loader = DataLoader(train_dataset, args.train_bs, True, pin_memory=True, num_workers=4, persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_dataset, args.test_bs, False, pin_memory=True, num_workers=4, persistent_workers=True, drop_last=False)

    logger.info("Building the clusters...")
    env_labels = build_clustering_model(train_dataset, args.disent_n_clusters, device=device)
    n_clusters = env_labels.max().item() + 1

    logger.info(f"Building the EgoZAR architecture for the following modalities {args.modality}...")
    model = {
        mod: EgoZAR(
            1024,
            train_dataset.env_feat_size(),
            n_segments=args.num_segments,
            hid_size=args.model_hidden_size,
            dropout=args.model_dropout,
            head_dropout=args.model_head_dropout,
            trn_dropout=args.model_trn_dropout,
            model_dom_classifier_hid_size=args.model_dom_classifier_hid_size,
            use_input_features=(args.use_input_features == "Y"),
            use_acz_features=(args.use_acz_features == "Y"),
            use_egozar_motion_features=args.ca and (args.use_egozar_motion_features == "Y"),
            use_egozar_acz_features=args.ca and (args.use_egozar_acz_features == "Y"),
            n_clusters=n_clusters,
        ).to(device)
        for mod in args.modality
    }
    parameters = [p for m in model.values() for p in m.parameters()]

    optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_steps, gamma=0.1)

    if args.use_wandb:
        for i, m in enumerate(model.values()):
            wandb.watch(m, log="all", idx=i, log_graph=True)

    logger.info("")
    logger.info(f"Starting the training loop for {args.num_epochs} epochs...")
    logger.info("")
    
    for epoch in range(1, 1 + args.num_epochs):
        logger.info(f"Starting epoch {epoch:3d}...")
        
        cls_loss, disent_loss = train(
            model,
            train_loader,
            optimizer,
            scheduler,
            double_ce=args.double_ce,
            disent_loss_weight=args.disent_loss_weight,
            disent_all_modalities=args.disent_all_modalities,
            env_labels=env_labels,
        )

        logger.info(f"Training losses: cls = {cls_loss:.3f}, disent = {disent_loss:.3f}.")

        log_dict = {"train/cls_loss": cls_loss, "train/disent_loss": disent_loss}

        if epoch % args.eval_freq == 0 or (args.num_epochs - epoch) < 10:
            (
                verb_top1,
                noun_top1,
                action_top1,
                verb_top5,
                noun_top5,
                action_top5,
            ) = validate(model, val_loader)

            avg_metric = ((verb_top1 + noun_top1 + action_top1) + (verb_top5 + noun_top5 + action_top5)) / 6

            logger.info("")
            logger.info("Target validation: ")
            logger.info(f" verb@1={verb_top1:.2f}, noun@1={noun_top1:.2f}, action@1={action_top1:.2f}")
            logger.info(f" verb@5={verb_top5:.2f}, noun@5={noun_top5:.2f}, action@5={action_top5:.2f}")
            logger.info(f" avg_metric={avg_metric:.2f}")

            log_dict.update(
                {
                    "val/target/loss": cls_loss,
                    "val/target/verb_top1": verb_top1,
                    "val/target/noun_top1": noun_top1,
                    "val/target/action_top1": action_top1,
                    "val/target/verb_top5": verb_top5,
                    "val/target/noun_top5": noun_top5,
                    "val/target/action_top5": action_top5,
                    "val/target/avg_metric": avg_metric,
                }
            )
            
        logger.info("")

        if args.use_wandb:
            wandb.log(log_dict, step=epoch)

    if args.save_model:
        path = osp.join(wandb.run.dir if args.use_wandb else ".", "checkpoint.pth")
        torch.save({mod: m.state_dict() for mod, m in model.items()}, path)

        if args.use_wandb:
            artifact = wandb.Artifact(name=args.save_model, type="model")
            artifact.add_file(local_path=path, name="checkpoint.pth")
            artifact.save()
    

if __name__ == "__main__":
    args = argparse.ArgumentParser("EgoZAR training")
    args.add_argument("--seed", default=1, type=int)
    
    # WandB arguments
    args.add_argument("--use-wandb", default="N", choices=["Y", "N"], help="Log training and validation metrics to wandb")
    args.add_argument("--wandb-project", type=str, help="WandB project name")
    args.add_argument("--wandb-entity", type=str, help="WandB project entity")
    
    # Generic training arguments
    args.add_argument("--modality", type=str, action="append", choices=["RGB", "Flow", "Audio"], help="Input modalities")
    args.add_argument("--num-segments", default=5, type=int, choices=[5, 10, 15, 20, 25], help="Number of input TBN segments")
    args.add_argument("--num-epochs", default=30, type=int, help="Number of training epochs")
    args.add_argument("--eval-freq", default=5, type=int, help="Validation every {arg} epochs")
    args.add_argument("--train-bs", default=128, type=int, help="Training batch size")
    args.add_argument("--test-bs", default=128, type=int, help="Validation batch size")
    args.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
    args.add_argument("--lr-steps", default=[10, 20], nargs="+", type=int, help="Learning rate steps")
    args.add_argument("--clip-features-prefix", default="clip_features/ViT-L_14_", choices=["clip_features/RN50_", "clip_features/ViT-L_14_"], 
                      help="Prefix for the zone recognition model features")

    # EgoZAR model arguments
    args.add_argument("--model-hidden-size", default=1024, type=int, )
    args.add_argument("--model-dropout", default=0.5, type=float)
    args.add_argument("--model-head-dropout", default=0.5, type=float)
    args.add_argument("--model-trn-dropout", default=0, type=float)
    args.add_argument("--model-dom-classifier-hid-size", default=1024, type=int)
    args.add_argument("--ca", action="store_true", default=False)
    args.add_argument("--use-input-features", default="Y", choices=["Y", "N"])
    args.add_argument("--use-egozar-motion-features", default="N", choices=["Y", "N"])
    args.add_argument("--use-egozar-acz-features", default="N", choices=["Y", "N"])
    args.add_argument("--use-acz-features", default="N", choices=["Y", "N"])

    # Training options
    args.add_argument("--double-ce", action="store_true", default=True)
    args.add_argument("--disent-loss-weight", type=float, default=0.0)
    args.add_argument("--disent-n-clusters", type=int, default=7)
    args.add_argument("--disent-all-modalities", action="store_true", default=False)

    args.add_argument("--save_model", action="store_true", default=False)

    args = args.parse_args()
    
    args.use_wandb = (args.use_wandb == "Y")

    if args.use_wandb:
        wandb.init(entity=args.wandb_project, project=args.wandb_entity)
        wandb.config.update(args)
    
    main(args)
        
    if args.use_wandb == "Y":
        wandb.finish()
    