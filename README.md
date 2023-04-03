# SAW

The official implementation OF paper "Sequence as A Whole: A Unified Framework for Video Action Localization with Long-range Text Query" \[[Paper](https://ieeexplore.ieee.org/document/10043827)\]

![](./docs/net.png)

We propose a unified framework which handles the whole video in sequential manner with long-range and dense visual-linguistic interaction in an end-to-end manner. Specifically, a lightweight relevance filtering based transformer (Ref-Transformer) is designed, which is composed of relevance filtering based attention and temporally expanded MLP. The text-relevant spatial regions and temporal clips in video can be efficiently highlighted through the relevance filtering and then propagated among the whole video sequence with the temporally expanded MLP. The unified framework can be utilized to varies video-text action localization tasks, e.g., referring video segmentation, temporal sentence grounding, and spatiotemporal video grounding. 

## Requirements

* python 3.8

* pytorch 1.9.1

* torchtext 0.10.1


## Referring Video Segmentation

run `cd referring_segmentation` for referring video segmentation task.

### 1. Dataset

Download the A2D Sentences dataset and J-HMDB Sentences dataset from [https://kgavrilyuk.github.io/publication/actor_action/](https://kgavrilyuk.github.io/publication/actor_action/) and convert the videos to RGB frames. 

For A2D Sentences dataset, run `python pre_proc\video2imgs.py` to convert videos to RGB frames. The following directory structure is expected:

```python
-a2d_sentences
    -Rename_Images
    -a2d_annotation_with_instances
    -videoset.csv
    -a2d_missed_videos.txt
    -a2d_annotation.txt
-jhmdb_sentences
    -Rename_Images
    -puppet_mask
    -jhmdb_annotation.txt
```

Edit the item `datasets_root` in `json/onfig_$DATASET$.json` to be the current dataset path.

Run `python pre_proc\generate_data_list.py` to generate the training and testing data splits.

### 2. Backbone

Download the pretrained DeepLabResNet from [https://github.com/VainF/DeepLabV3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch) and put it into `model/pretrained/`. 

### 4. Training

Only the A2D Sentences dataset is adopted for training, run:

```python
python main.py --json_file=json\config_a2d_sentences.json --mode=train
```

### 5. Evaluation

For A2d Sentences dataset, run:

```python
python main.py --json_file=json\config_a2d_sentences.json --mode=test
``` 

For JHMDB Sentences dataset, run:

```python
python main.py --json_file=json\config_jhmdb_sentences.json --mode=test
``` 

## Temporal Sentence Grounding

run `cd temporal_grounding` for referring temporal sentence grounding task.

### 1. Dataset

* For charades-STA dataset, download the pre-extracted I3D features following [LGI4temporalgrounding](https://github.com/JonghwanMun/LGI4temporalgrounding) and the pre-extracted VGG feature following [2D-TAN](https://github.com/microsoft/VideoX/tree/master/2D-TAN).

* For TACoS dataset, download the pre-extracted C3D features following [2D-TAN](https://github.com/microsoft/VideoX/tree/master/2D-TAN)

* For ActivityNet Captions dataset, download the pre-extracted C3D features from [http://activity-net.org/challenges/2016/download.html](http://activity-net.org/challenges/2016/download.html).

### 2. Training and Evaluation

The config files can be find in `./json` and the following model settings are supported

```
-config_ActivityNet_C3D_anchor.json
-config_ActivityNet_C3D_regression.json
-config_Charades-STA_I3D_anchor.json
-config_Charades-STA_I3D_regression.json
-config_Charades-STA_VGG_anchor.json
-config_Charades-STA_VGG_regression.json
-config_TACoS_C3D_anchor.json
-config_TACoS_C3D_regression.json
```

Set the `"datasets_root"` in each config file to be your feature path.

To train on different dataset with different grounding heads, run

```
python main.py --json_file=$JSON_FILE_PATH$ --mode=train
```

For evaluation, run 

```
python main.py --json_file=$JSON_FILE_PATH$ --mode=test --checkpoint=$CHECKPOINT_PATH$
```

The pretrained models and their correspondance performance are shown bellow

| Datasets     | Feature | Decoder    | Checkpoints |
|--------------|---------|------------|-------------|
| Charades-STA | I3D     | Regression        |  \[[Baidu](https://pan.baidu.com/s/1GQBkElQITd-exS1njNZrwQ) \| gj54 \]           |
| Charades-STA | I3D     | Anchor     |    \[[Baidu](https://pan.baidu.com/s/1MXZqAEBLOzauR8cOLjo3QA) \| 5j3a \]           |
| Charades-STA | VGG     | Regression |                \[[Baidu](https://pan.baidu.com/s/1Yacke_tkaAELzMY_ePyIhw) \| 52xf \]           |
| Charades-STA | VGG     | Anchor     |                \[[Baidu](https://pan.baidu.com/s/1PcIZ7QEWcYnfzne1dkMsng) \| rdmx \]          |
| ActivityNet  | C3D     | Regression |                \[[Baidu](https://pan.baidu.com/s/1zlH64seHimscTOtNry-6Ag) \| 6sbh \]            |
| ActivityNet  | C3D     | Anchor     |              \[[Baidu](https://pan.baidu.com/s/1mi8M2wBUAqskWQQqHdmi2Q) \| ysr5 \]           |
| TACOS        | C3D     | Regression |                \[[Baidu](https://pan.baidu.com/s/140m-9geYbktSRfP7Pa1rzA) \| iwx2 \]           |
| TACOS        | C3D     | Anchor     |               \[[Baidu](https://pan.baidu.com/s/1dzIIb4dKQY9t-oAF-N2sLw) \| 1ube \]           |


## Spatiotemporal Video Grounding

run `cd spatiotemporal_grounding` for spatiotemporal video grounding task. The code for spatiotemporal grounding is built on the [TubeDETR codebase](https://github.com/antoyang/TubeDETR).

### 1. Dataset

We prepare the `HC-STVG` and `VidSTG` datasets following the [TubeDETR](https://github.com/antoyang/TubeDETR). The annotation formation of the VidSTG dataset has been optimized to reduce the training memory usage. 

**videos**

VidSTG dataset: Download VidOR videos from [the VidOR dataset providers](https://xdshang.github.io/docs/vidor.html)

HC-STVG dataset: Download HC-STVG videos from [the HC-STVG dataset providers](https://github.com/tzhhhh123/HC-STVG).

Edit the item `vidstg_vid_path` in `spatiotemporal_grounding/config/vidstg.json` and the `hcstvg_vid_path` in `spatiotemporal_grounding/config/hcstvg.json` to be the current video path.

**annotations**

Download the preprocessed annotation files from \[[https://pan.baidu.com/s/1oiV9PmtRqRxxdxMvqrJj_w](https://pan.baidu.com/s/1oiV9PmtRqRxxdxMvqrJj_w), password: n6y4\]. Then put the downloaded `annotations` into `spatiotemporal_grounding`.

### 2. Training and Evaluation

To train on HC-STVG dataset, run

```
python main.py --combine_datasets=hcstvg --combine_datasets_val=hcstvg --dataset_config config/hcstvg.json --output-dir=hcstvg_result
```

To train on VidSTG dataset, run

```
python main.py --combine_datasets=vidstg --combine_datasets_val=vidstg --dataset_config config/vidstg.json --output-dir=vidstg_result
```

To evaluate on HC-STVG dataset, run:

```
python main.py --combine_datasets=hcstvg --combine_datasets_val=hcstvg --dataset_config config/hcstvg.json --output-dir=hcstvg_result --eval --resume=$CHECKPOINT_PATH$
```

To evaluate on VidSTG dataset, run

```
python main.py --combine_datasets=vidstg --combine_datasets_val=vidstg --dataset_config config/vidstg.json --output-dir=vidstg_result --eval --resume=$CHECKPOINT_PATH$
```

## Citation

```
@article{2023saw,
    title     = {Sequence as A Whole: A Unified Framework for Video Action Localization with Long-range Text Query},
    author    = {Yuting Su, Weikang Wang, Jing Liu, Shuang Ma, Xiaokang Yang},
    booktitle = {IEEE Transactions on Image Processing},
    year      = {2023}
}
```