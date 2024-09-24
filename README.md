# Egocentric zone-aware action recognition across environments

[Simone Alberto Peirone](https://scholar.google.com/citations?user=K0efPssAAAAJ)\*, [Gabriele Goletto](https://gabrielegoletto.github.io)\*, [Mirco Planamente](https://scholar.google.com/citations?user=GIJ3h4AAAAAJ), [Andrea Bottino](https://scholar.google.com/citations?user=YWhB9iYAAAAJ), [Barbara Caputo](https://scholar.google.com/citations?user=mHbdIAwAAAAJ&hl=en), [Giuseppe Averta](https://scholar.google.com/citations?user=i4rm0tYAAAAJ)

<a href='https://arxiv.org/abs/2409.14205'><img src='https://img.shields.io/badge/Paper-Arxiv:2409.14205-red'></a>
<a href='https://gabrielegoletto.github.io/EgoZAR/'><img src='https://img.shields.io/badge/Project-Page-Green'></a><a target="_blank" href="https://colab.research.google.com/github/sapeirone/EgoZAR/blob/main/run.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


This is the official PyTorch implementation of our work "Egocentric zone-aware action recognition across environments".

**Abstract**:

Human activities exhibit a strong correlation between actions and the places where these are performed, such as washing something at a sink. 
More specifically, in daily living environments we may identify particular locations, hereinafter named *activity-centric zones*, which may afford a set of homogeneous actions. 
Their knowledge can serve as a prior to favor vision models to recognize human activities.


However, the appearance of these zones is scene-specific, limiting the transferability of this prior information to unfamiliar areas and domains. This problem is particularly relevant in egocentric vision, where the environment takes up most of the image, making it even more difficult to separate the action from the context.
In this paper, we discuss the importance of decoupling the domain-specific appearance of activity-centric zones from their universal, domain-agnostic representations, and show how the latter can improve the cross-domain transferability of Egocentric Action Recognition (EAR) models. 
We validate our solution on the EPIC-Kitchens-100 and Argo1M datasets.

## Getting Started

### 1. Clone the Repository and Set Up Environment
Clone this repository and create a Conda environment:
```sh
git clone --recursive https://github.com/sapeirone/EgoZAR
cd EgoZAR
conda create --name egozar
conda activate egozar
pip install -r requirements.txt
```

### 2. Download the EK100 Annotations and TBN Features
The official EK100 UDA annotations and pre-extracted features are provided [here](https://github.com/epic-kitchens/C4-UDA-for-Action-Recognition).

```sh
# Annotations
mkdir -p annotations
git clone https://github.com/epic-kitchens/epic-kitchens-100-annotations.git
mv epic-kitchens-100-annotations/UDA_annotations EgoZAR/annotations
rm -r epic-kitchens-100-annotations

# TBN features
mkdir -p data
wget -O ek100.zip https://www.dropbox.com/scl/fo/us8zy3r2rufqriig0pbii/ABeUdV83UNmJ5US-oCxAPno?rlkey=yzbuczl198z067pnotx1zxvuo&e=1&dl=0
unzip ek100.zip -d data/
rm ek100.zip
```

### 3. Extract the CLIP features for EgoZAR
Optional: for easy prototyping you can download the pre-extracted CLIP ViT-L/14 features for the [source train]() and [target validation]() splits.

#### 3.1 Download the EPIC-Kitchens RGB frames
Download the EPIC-Kitchens RGB frames under the `EPIC-KITCHENS` directory, following the [official instructions](https://github.com/epic-kitchens/epic-kitchens-download-scripts).

The expected data stucture for EPIC-KITCHENS videos is:
```text
│
├── EPIC-KITCHENS/
│   ├── <p_id>/
│   │   ├── rgb_frames/
│   │   │   └── <video_id>/
│   │   │       ├── frame_0000000000.jpg
│   │   │       ├── frame_0000000001.jpg
│   │   │       └── ...
│   │
│   └── ...
│
└── ...
```

#### 3.2 Extract the CLIP features
Extract the CLIP features using the `save_CLIP_features.py` script for the desired CLIP variant.

```sh
mkdir -p clip_features
python save_CLIP_features.py --clip-model=ViT-L/14
```

This command should generate the files `clip_features/ViT-L_14_source_train.pth` and `clip_features/ViT-L_14_target_val.pth` for source train and target validation respectively.


#### (Optional) 4. Train the multimodal baseline
Train the *Source Only* multimodal baseline with the following command:
```
python train.py --modality=RGB --modality=Flow --modality=Audio
```

#### 5. Train EgoZAR using ViT-L/14 features
```sh
python train.py --modality=RGB --modality=Flow --modality=Audio --ca \
    --use-input-features=N --use-egozar-motion-features=Y --use-egozar-acz-features=Y \
    --disent-loss-weight=1.0 \
    --disent-n-clusters=4
```

## Acknowledgements
This study was supported in part by the CINI Consortium through the VIDESEC project and carried out within the FAIR - Future Artificial Intelligence Research and received funding from the European Union Next-GenerationEU (PIANO NAZIONALE DI RIPRESA E RESILIENZA (PNRR) – MISSIONE 4 COMPONENTE 2, INVESTIMENTO 1.3 – D.D. 1555 11/10/2022, PE00000013). This manuscript reflects only the authors’ views and opinions, neither the European Union nor the European Commission can be considered responsible for them. G. Goletto is supported by PON “Ricerca e Innovazione” 2014-2020 – DM 1061/2021 funds.

## Cite Us
If you use EgoZAR in your research or applications, please cite our paper:
```bibtex
@article{peirone2024,
  author    = {Peirone, Simone Alberto and Goletto, Gabriele and Planamente, Mirco and Bottino, Andrea and Caputo, Barbara and Averta, Giuseppe},
  title     = {Egocentric zone-aware action recognition across environments},
  journal   = {arXiv preprint arXiv:2409.14205},
  year      = {2024},
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
