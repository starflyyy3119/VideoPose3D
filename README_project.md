# Dependencies
- Python 3+ distribution (need to set up on gpu farm)
- [Pytorch](https://pytorch.org/get-started/locally/) >= 0.4.0
(need to set up on the gpu farm)
# Evaluating the pretrainied model
- Preprocess the Human3.6 DataSet and get npz files(done!, download [data_2d_h36m_cpn_ft_h36m_dbb.npz](https://drive.google.com/file/d/1FfnFpFzoOsJ2kzaY_L9q8buS7t2GClBL/view?usp=sharing) and put it in the ```data``` folder)
- Download ```pretrained_h36m_cpn.bin``` and put it into the ```checkpoint``` folder. (done!)
- Test on Human 3.6m. (done!)

```
python run.py -k cpn_ft_h36m_dbb -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_cpn.bin
```

# Run on custom videos
## 1. Set up
- Use ```pip install ffmpeg``` to install the specific library. (need to set up on gpu farm)
- Download the [pre-trainined model](https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin) and put it into the checkpoint direcotry.(done, the file name is ```pretrained_h36m_detectron_coco.bin```)

## 2. Video Preprocessing
- Prepare a video and put it into video folder.(done!)
- Use ffmpeg library to cut the specific part of the video.(done!, the preprocessed video is ```cxk.mp4``` in the ```video``` directory)

```shell
ffmpeg -i demo.mp4 -ss 0:11 -to 0:49 -c copy cxk.mp4
```

## 3. Inferring 2D keypoints with Detectectron
- Set up [Detectron2](https://github.com/facebookresearch/detectron2) and infer 2D keypoints. (need to set up on gpu farm)
**To Do** (Since I'm not able to work on gpu farm) Follow the ```Inference.md``` and make visualizations.

```shell
cd inference
python infer_video_d2.py \
    --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml \
    --output-dir video_output \
    --image-ext mp4 \
    video
```



