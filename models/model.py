import torch
import torch.nn as nn

from einops import rearrange

from typing import Optional

from models.GRL import GradientReversalLayer
from models.cross_attn import CrossAttnLayer


class EgoZAR(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        in_channels_clip: int,
        n_segments: int = 5,
        hid_size: int = 512,
        dropout: float = 0.5,
        head_dropout: float = 0.5,
        trn_dropout: float = 0,
        *args,
        use_input_features: bool = True,
        use_acz_features: bool = False,
        use_egozar_motion_features: bool = False,
        use_egozar_acz_features: bool = False,
        model_dom_classifier_hid_size: int = 64,
        n_clusters: int = 0,
        **kwargs,
    ) -> None:
        """EgoZAR model.

        Parameters
        ----------
        in_channels : int
            size of the action recognition features
        in_channels_clip : int
            size of zone recognition model features
        n_segments : int, optional
            number of input segments, by default 5
        hid_size : int, optional
            internal hidden size of the model, by default 512
        dropout : float, optional
            dropout in the projection layers, by default 0.5
        head_dropout : float, optional
            dropout in the classification heads, by default 0.5
        trn_dropout : float, optional
            dropout in the trn layer, by default 0
        use_input_features : bool, optional
            if True projected features are fed to the trn module, by default True
        use_acz_features : bool, optional
            if True the features of the zone extraction model are fed to the trn module, by default False
        use_egozar_motion_features : bool, optional
            if True the output features of the action extraction module are fed to the trn module, by default False
        use_egozar_acz_features : bool, optional
            if True the output features of the zone extraction module are fed to the trn module, by default False
        model_dom_classifier_hid_size : int, optional
            hidden size of the adversarial classifiers, by default 64
        n_clusters : int, optional
            number of zone clusters, by default 0
        """
        super().__init__(*args, **kwargs)
        self.n_segments = n_segments

        # Projection layers
        self.backbone_mod = self._build_proj(in_channels, hid_size, hid_size, dropout)
        self.backbone_clip = self._build_proj(in_channels_clip, hid_size, hid_size, dropout)
        
        # Attention-based interaction modules
        self.interact_ca = CrossAttnLayer(embed_dimension=hid_size)
        self.interact_sa_clip = CrossAttnLayer(embed_dimension=hid_size)
        self.interact_sa = CrossAttnLayer(embed_dimension=hid_size)

        self.use_input_features = use_input_features
        self.use_acz_features = use_acz_features
        self.use_egozar_motion_features = use_egozar_motion_features
        self.use_egozar_acz_features = use_egozar_acz_features
        trn_expansion_factor = use_input_features + use_acz_features + use_egozar_acz_features + use_egozar_motion_features

        self.trn = nn.Sequential(
            nn.Dropout(trn_dropout),
            nn.ReLU(),
            nn.Linear(self.n_segments * hid_size * trn_expansion_factor, hid_size),
            nn.ReLU(),
        )

        self.verb_classifier = self._build_proj(hid_size, hid_size, 97, head_dropout)
        self.noun_classifier = self._build_proj(hid_size, hid_size, 300, head_dropout)
        
        self.n_clusters = n_clusters

        self.cls_acz = nn.Sequential(
            nn.Linear(hid_size, model_dom_classifier_hid_size),
            nn.ReLU(),
            nn.Linear(model_dom_classifier_hid_size, n_clusters),
        )
        self.cls_motion = nn.Sequential(
            GradientReversalLayer(),
            nn.Linear(hid_size, model_dom_classifier_hid_size),
            nn.ReLU(),
            nn.Linear(model_dom_classifier_hid_size, n_clusters, bias=False),
        )

    def _build_proj(self, in_channels, hidden_size, out_channels, dropout=0):
        return nn.Sequential(
            nn.Linear(in_channels, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, out_channels),
        )

    def forward(self, x_mod: torch.Tensor, x_acz: torch.Tensor, env_labels: Optional[torch.Tensor]):

        # Step 1: project the modality and acz features
        x_mod = rearrange(x_mod, "bs n h -> (bs n) h")
        x_acz = rearrange(x_acz, "bs n h -> (bs n) h")

        x_mod = self.backbone_mod(x_mod)
        x_acz = self.backbone_clip(x_acz)

        # Step 2: cross attention mod/acz + self attention acz
        x_mod = rearrange(x_mod, "(bs n) h -> bs n h", n=self.n_segments)
        x_acz = rearrange(x_acz, "(bs n) h -> bs n h", n=self.n_segments)
        feat_motion = self.interact_ca(x_acz, x_mod)
        feat_acz = self.interact_sa_clip(x_acz, x_acz)

        # Step 3: adversarial disentaglement of motion and acz features
        motion_cls, acz_cls = None, None
        if env_labels is not None:
            motion_cls = self.cls_motion(feat_motion.mean(1))
            acz_cls = self.cls_acz(feat_acz.mean(1))

        # Step 4: concatenate all features and pass through TRN
        x = []
        if self.use_input_features:
            x.append(x_mod)
        if self.use_acz_features:
            x.append(x_acz)
        if self.use_egozar_motion_features:
            x.append(feat_motion)
        if self.use_egozar_acz_features:
            x.append(feat_acz)

        x = torch.cat(x, -1)
        x = rearrange(x, "bs n h -> bs (n h)", n=self.n_segments)
        x = self.trn(x)

        out_verb = self.verb_classifier(x)
        out_noun = self.noun_classifier(x)

        return (
            # classifier outputs
            out_verb,
            out_noun,
            # affordance and motion features
            feat_motion,
            feat_acz,
            # assignment classifiers
            motion_cls,
            acz_cls,
        )
