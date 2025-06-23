# MedSAM
This is the official repository for MedSAM: Segment Anything in Medical Images.

Welcome to join our [mailing list](https://forms.gle/hk4Efp6uWnhjUHFP6) to get updates.


## News

- 2024.08.06: Med[SAM2](https://github.com/facebookresearch/segment-anything-2)-Segment Anything in Medical Images and Videos: Benchmark and Deployment [[`Paper`](https://arxiv.org/abs/2408.03322)] [[`Code`](https://github.com/bowang-lab/MedSAM/tree/MedSAM2)] [[Online Demo](https://huggingface.co/spaces/junma/MedSAM2)] [[`Gradio API`](https://github.com/bowang-lab/MedSAM/blob/MedSAM2/app.py)] [[`3D Slicer Plugin`](https://github.com/bowang-lab/MedSAMSlicer/tree/SAM2)] [[Fine-tune SAM2](https://github.com/bowang-lab/MedSAM/tree/MedSAM2?tab=readme-ov-file#fine-tune-sam2-on-the-abdomen-ct-dataset)]
- 2024.01.15: Welcome to join [CVPR 2024 Challenge: MedSAM on Laptop](https://www.codabench.org/competitions/1847/)
- 2024.01.15: Release [LiteMedSAM](https://github.com/bowang-lab/MedSAM/blob/LiteMedSAM/README.md) and [3D Slicer Plugin](https://github.com/bowang-lab/MedSAMSlicer), 10x faster than MedSAM! 


## Installation
1. Create a virtual environment `conda create -n medsam python=3.10 -y` and activate it `conda activate medsam`
2. Install [Pytorch 2.0](https://pytorch.org/get-started/locally/)
3. `git clone https://github.com/bowang-lab/MedSAM`
4. Enter the MedSAM folder `cd MedSAM` and run `pip install -e .`


## Get Started
Download the [model checkpoint](https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN?usp=drive_link) and place it at e.g., `work_dir/MedSAM/medsam_vit_b`

We provide three ways to quickly test the model on your images

1. Command line

```bash
python MedSAM_Inference.py # segment the demo image
```

Segment other images with the following flags
```bash
-i input_img
-o output path
--box bounding box of the segmentation target
```

2. Jupyter-notebook

We provide a step-by-step tutorial on [CoLab](https://colab.research.google.com/drive/19WNtRMbpsxeqimBlmJwtd1dzpaIvK2FZ?usp=sharing)

You can also run it locally with `tutorial_quickstart.ipynb`.

3. GUI

Install `PyQt5` with [pip](https://pypi.org/project/PyQt5/): `pip install PyQt5 ` or [conda](https://anaconda.org/anaconda/pyqt): `conda install -c anaconda pyqt`

```bash
python gui.py
```

Load the image to the GUI and specify segmentation targets by drawing bounding boxes.



https://github.com/bowang-lab/MedSAM/assets/19947331/a8d94b4d-0221-4d09-a43a-1251842487ee





## Model Training

### Data preprocessing

Download [SAM checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) and place it at `work_dir/SAM/sam_vit_b_01ec64.pth` .

Download the demo [dataset](https://zenodo.org/record/7860267) and unzip it to `data/FLARE22Train/`.

This dataset contains 50 abdomen CT scans and each scan contains an annotation mask with 13 organs. The names of the organ label are available at [MICCAI FLARE2022](https://flare22.grand-challenge.org/).

Run pre-processing

Install `cc3d`: `pip install connected-components-3d`

```bash
python pre_CT_MR.py
```

- split dataset: 80% for training and 20% for testing
- adjust CT scans to [soft tissue](https://radiopaedia.org/articles/windowing-ct) window level (40) and width (400)
- max-min normalization
- resample image size to `1024x1024`
- save the pre-processed images and labels as `npy` files


### Training on multiple GPUs (Recommend)

The model was trained on five A100 nodes and each node has four GPUs (80G) (20 A100 GPUs in total). Please use the slurm script to start the training process.

```bash
sbatch train_multi_gpus.sh
```

When the training process is done, please convert the checkpoint to SAM's format for convenient inference.

```bash
python utils/ckpt_convert.py # Please set the corresponding checkpoint path first
```

### Training on one GPU

```bash
python train_one_gpu.py
```



## Acknowledgements
- We highly appreciate all the challenge organizers and dataset owners for providing the public dataset to the community.
- We thank Meta AI for making the source code of [segment anything](https://github.com/facebookresearch/segment-anything) publicly available.
- We also thank Alexandre Bonnet for sharing this great [blog](https://encord.com/blog/learn-how-to-fine-tune-the-segment-anything-model-sam/)


## 心脏运动追踪模块

我们新增了一个心脏运动追踪模块，专门用于处理心脏视频序列。该模块能够：

1. 使用轻量级卷积网络检测视频帧间的运动区域（例如心脏瓣膜）
2. 为每一帧生成运动热力图
3. 基于运动热力图自动分割心脏结构
4. 在视频上实时追踪运动区域

### 使用方法

```bash
python cardiac_demo.py --input [视频文件或图像序列文件夹] --output [输出目录]
```

参数说明：
- `--input`: 输入视频文件路径或包含视频帧的文件夹路径
- `--output`: 输出结果保存路径（默认: ./output）
- `--sam_checkpoint`: MedSAM模型检查点路径（默认: work_dir/MedSAM/medsam_vit_b.pth）
- `--model_type`: SAM模型类型（默认: vit_b）
- `--device`: 使用的设备（默认: cuda:0）
- `--motion_threshold`: 运动热力图阈值，大于此值的区域被视为运动区域（默认: 0.3）

### 输出文件

该模块会生成以下输出：
- `motion_heatmap.mp4`: 运动热力图视频
- `segmentation_overlay.mp4`: 分割结果叠加视频
- 关键帧的热力图和分割结果图像

### 在您的项目中使用

您可以在自己的代码中引入心脏运动追踪模块：

```python
from segment_anything import sam_model_registry
from cardiac_motion_tracker import CardiacMotionTracker

# 加载MedSAM模型
sam_model = sam_model_registry["vit_b"](checkpoint="path/to/checkpoint")
sam_model.to(device)
sam_model.eval()

# 创建心脏运动追踪器
motion_tracker = CardiacMotionTracker(
    sam_model=sam_model,
    motion_threshold=0.3
)

# 处理单帧
result = motion_tracker.process_frame(frame_tensor)  # frame_tensor: [B, 3, H, W]

# 处理整个视频
results = motion_tracker.process_video(video_frames)  # video_frames: list of [3, H, W] tensors
```

## Reference

```
@article{MedSAM,
  title={Segment Anything in Medical Images},
  author={Ma, Jun and He, Yuting and Li, Feifei and Han, Lin and You, Chenyu and Wang, Bo},
  journal={Nature Communications},
  volume={15},
  pages={654},
  year={2024}
}
```
