# SAW

The code for paper "Sequence as A Whole: A Unified Framework for Video Action Localization with Long-range Text Query"

![](./docs/net.png)

We propose a unified framework which handles the whole video in sequential manner with long-range and dense visual-linguistic interaction in an end-to-end manner. Specifically, a lightweight relevance filtering based transformer (Ref-Transformer) is designed, which is composed of relevance filtering based attention and temporally expanded MLP. The text-relevant spatial regions and temporal clips in video can be efficiently highlighted through the relevance filtering and then propagated among the whole video sequence with the temporally expanded MLP. The unified framework can be utilized to varies video-text action localization tasks, e.g., referring video segmentation, temporal sentence grounding, and spatiotemporal video grounding. 

## Requirements

* python 3.8

* pytorch 1.9.1

* torchtext 0.10.1

* flair 0.11.1

## Referring Segmentation

### 1. Dataset

Download the A2D Sentences dataset and J-HMDB Sentences dataset from [https://kgavrilyuk.github.io/publication/actor_action/](https://kgavrilyuk.github.io/publication/actor_action/) and convert the videos to RGB frames. The following directory structure is expected:
```python
-A2D
    -Rename_Images
    -a2d_annotation_with_instances
-JHMDB
    -Rename_Images
    -puppet_mask
```
Edit the item `datasets_root` in `json/config.json` to be the current dataset path.

### 2. Word Embedding

Download the Glove embedding from [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/) and put it into `model/pretrained/`.

### 3. Backbone

Download the pretrained DeepLabResNet from [https://github.com/VainF/DeepLabV3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch) and put it into `model/pretrained/`. 

### 4. Training

Set the item `"mode"` to `"train“` in `json/config.json` and then please run:
```
python main.py
```
### 5. Evaluation

Set the item `"mode"` to `"test“` in `json/config.json` and then please run:
```
python main.py
``` 

## Temporal Sentence Grounding

