import torch
import torch.nn as nn

from einops import rearrange

from models.GRL import GradientReversalLayer
from models.cross_attn import CrossAttnLayer


class EgoZAR(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        in_channels_clip,
        n_segments=5,
        hid_size=512,
        dropout=0.5,
        head_dropout=0.5,
        trn_dropout: float = 0,
        *args,
        use_input_features: bool = True,
        use_acz_features: bool = False,
        use_egozar_motion_features: bool = False,
        use_egozar_acz_features: bool = False,
        model_dom_classifier_hid_size: int = 64,
        normalize_before_trn: bool = False,
        n_clusters: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.n_segments = n_segments

        self.backbone_mod = self._build_proj(in_channels, hid_size, hid_size, dropout)
        self.backbone_clip = self._build_proj(in_channels_clip, hid_size, hid_size, dropout)
        self.interact_ca = CrossAttnLayer(embed_dimension=hid_size)
        self.interact_sa_clip = CrossAttnLayer(embed_dimension=hid_size)
        self.interact_sa = CrossAttnLayer(embed_dimension=hid_size)

        self.use_input_features = use_input_features
        self.use_acz_features = use_acz_features
        self.use_egozar_motion_features = use_egozar_motion_features
        self.use_egozar_acz_features = use_egozar_acz_features
        trn_expansion_factor = use_input_features + use_acz_features + use_egozar_acz_features + use_egozar_motion_features

        self.norm_inp = nn.LayerNorm(hid_size) if normalize_before_trn else nn.Identity()
        self.norm_motion = nn.LayerNorm(hid_size) if normalize_before_trn else nn.Identity()
        self.norm_acz = nn.LayerNorm(hid_size) if normalize_before_trn else nn.Identity()
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
        
        self.cls_domain = nn.Sequential(
                GradientReversalLayer(),
                nn.Linear(hid_size, model_dom_classifier_hid_size),
                nn.ReLU(),
                nn.Linear(model_dom_classifier_hid_size, 12),
            )

    def _build_shallow_proj(self, in_channels, out_channels):
        return nn.Linear(in_channels, out_channels, bias=True)

    def _build_proj(self, in_channels, hidden_size, out_channels, dropout=0):
        return nn.Sequential(
            nn.Linear(in_channels, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, out_channels),
        )

    def forward(self, x_mod, x_acz, env_labels=None):
        # x has shape bs x n x h

        # Step 1: project the modality and acz features
        x_mod = rearrange(x_mod, "bs n h -> (bs n) h")
        x_acz = rearrange(x_acz, "bs n h -> (bs n) h")

        x_mod = self.backbone_mod(x_mod)
        x_acz = self.backbone_clip(x_acz)

        # Step 2: cross attention mod/acz + self attention acz
        x_mod = rearrange(x_mod, "(bs n) h -> bs n h", n=self.n_segments)
        x_acz = rearrange(x_acz, "(bs n) h -> bs n h", n=self.n_segments)
        # feat_motion = self.proj_motion(self.interact_ca(x_acz, x_mod))
        # feat_acz = self.proj_acz(self.interact_sa_clip(x_acz, x_acz))
        feat_motion = self.interact_ca(x_acz, x_mod)
        feat_acz = self.interact_sa_clip(x_acz, x_acz)

        # Step 3: adversarial disentaglement of motion and acz features
        motion_cls, acz_cls = None, None
        if env_labels is not None:
            motion_cls = self.cls_motion(feat_motion.mean(1))
            acz_cls = self.cls_acz(feat_acz.mean(1))
        domain_cls = self.cls_domain(feat_motion.mean(1))

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
