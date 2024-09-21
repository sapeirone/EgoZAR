import os.path as osp

from collections import namedtuple

import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
from PIL import Image

from typing import List, Literal, Dict, Tuple

import logging
logger = logging.getLogger(__name__)


Record = namedtuple("record", ["segment_id", "video_id", "participant_id",
                               "start_frame", "stop_frame", "verb_class",
                               "noun_class", "narration"])

SplitType = Literal['source', 'target']
ModeType = Literal['train', 'val']
ModalityType = Literal['RGB', 'Flow', 'Audio']


class EK100Dataset(Dataset):

    def __init__(self, split: SplitType, mode: ModeType, 
                 modalities: List[ModalityType],
                 root: str = 'EPIC-KITCHENS', 
                 transform=None, n: int = 5,
                 num_segments=25, return_frames: bool = True,
                 clip_features_path=None) -> None:
        """Video Dataset for the EK100 dataset.
        
        This dataset returns the preextracted TBN and CLIP features for the EK100 UDA benchmark.

        Parameters
        ----------
        split : SplitType
            domain, either source or target
        mode : ModeType
            mode, either train or val
        modalities : List[ModalityType]
            list of modalities to be returned
        root: str
            root path of the EPIC-Kitchens-100 data
        transform : optional
            transform applied to raw frames, by default None
        n : int, optional
            number of sampled segments, by default 5
        num_segments : int, optional
            number of segments for action sample, by default 25
        return_frames : bool, optional
            whether to return raw frames (True) or the corresponding CLIP features (False), by default True
        root: str
        clip_features_path : _type_, optional
            path to the pre-extracted CLIP features, by default None
        """
        super().__init__()
        
        logger.info(f"Loading EK100 dataset for split {split} and mode {mode}.")

        self.n: int = n
        self.num_segments: int = num_segments
        
        self.root = root

        self.split: SplitType = split
        self.mode: ModeType = mode
        self.return_frames: bool = return_frames
        
        self.transform = transform

        if not osp.exists(f"annotations/EPIC_100_uda_{split}_{mode}.csv"):
            raise ValueError(f"Annotations for split {split} and mode {mode} not found!")
        
        # Load the input features
        annotations: pd.DataFrame = pd.read_csv(f"annotations/EPIC_100_uda_{split}_{mode}.csv")
        self.features: Dict[str, np.array] = {
            modality: features 
            for modality, features in pd.read_pickle(f"data/{split}_{mode}.pkl")['features'].items() if modality in modalities
        }

        # If self.return_frames is False, this dataset returns raw frames rather than the CLIP features
        if not self.return_frames:
            assert clip_features_path is not None
            self.clip_features = torch.load(clip_features_path, weights_only=True)

        # Load the action samples
        self.data: List[Record] = [
            Record(idx, a.video_id, a.participant_id, a.start_frame, a.stop_frame, a.verb_class, a.noun_class, a.narration)
            for idx, (_, a) in enumerate(annotations.iterrows())
        ]

    def env_feat_size(self) -> int:
        """Get the size of the pre-extracted CLIP features.

        Returns
        -------
        int
            CLIP features size

        Raises
        ------
        ValueError
            if the dataset did not load the pre-extracted CLIP features.
        """
        if not hasattr(self, 'clip_features'):
            raise ValueError("This dataset is not loading CLIP features. Please initialize the dataset with return_frames = False")
        
        return self.clip_features.shape[-1]

    def __len__(self) -> int:
        """Return the number of actions in the dataset.

        Returns
        -------
        int
            number of samples in the dataset
        """
        return len(self.data)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], int, int, int]:
        """Return one sample from the dataset.

        Parameters
        ----------
        i : int
            index of the sample to return

        Returns
        -------
        Tuple[torch.Tensor, Dict[str, torch.Tensor], int, int]
            sample data
        """
        data = self.data[i]
        clip_frames: np.array = np.linspace(data.start_frame, data.stop_frame, num=self.n, dtype=int)

        tick = self.num_segments / self.n
        features_indices = [1 + int(tick / 2.0 + tick * x) for x in range(self.n)]

        features = {
            modality: torch.stack([
                torch.from_numpy(self.features[modality][data.segment_id][idx]).float()
                for idx in features_indices
            ])
            for modality in self.features.keys()
        }

        if self.return_frames:
            # Return raw frames
            frames = torch.stack([
                self.transform(Image.open(f"{self.root}/{data.participant_id}/rgb_frames/{data.video_id}/frame_{frame:010d}.jpg"))
                for frame in clip_frames
            ])
        else:
            # Return pre-extracted CLIP features
            frames = self.clip_features[i]

        return frames, features, data.verb_class, data.noun_class, i
    
    @staticmethod
    def build_dataset(domain, split, modalities, num_segments=5, clip_features_prefix=None):
        return EK100Dataset(
            domain,
            split,
            modalities=modalities,
            n=num_segments,
            num_segments=25,
            return_frames=False,
            clip_features_path=f"{clip_features_prefix}{domain}_{split}.pth",
        )
