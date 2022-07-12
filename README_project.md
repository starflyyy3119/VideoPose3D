# Dependencies

- Python 3+ distribution
- [Pytorch](https://pytorch.org/get-started/previous-versions/) v1.10.0 (for compatibility with Detectron2)
- Cuda v11.3 (for compatibility with Detectron2)

# Evaluating the pretrainied model

- Preprocess the Human3.6 DataSet and get npz files(download [data_2d_h36m_cpn_ft_h36m_dbb.npz](https://drive.google.com/file/d/1FfnFpFzoOsJ2kzaY_L9q8buS7t2GClBL/view?usp=sharing) and [data_3d_h36m.npz](https://drive.google.com/file/d/1mFAlUkeIUCguSvYLib21l6VY6ByVFsQk/view?usp=sharing) put it in the `data` folder)
- Download [pretrained_h36m_cpn.bin] and put it into the `checkpoint` folder.
- Test on Human 3.6m.

```
!gdown --fuzzy https://drive.google.com/file/d/1FfnFpFzoOsJ2kzaY_L9q8buS7t2GClBL/view?usp=sharing -O data/data_2d_h36m_cpn_ft_h36m_dbb.npz
!gdown --fuzzy https://drive.google.com/file/d/1mFAlUkeIUCguSvYLib21l6VY6ByVFsQk/view?usp=sharing -O data/data_3d_h36m.npz
!wget -c https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_cpn.bin -O checkpoint/pretrained_h36m_cpn.bin
python run.py -k cpn_ft_h36m_dbb -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_cpn.bin
```

# Run on custom videos

## 1. Set up

- Use `pip install ffmpeg` to install the specific library. (need to set up on gpu farm)
- Download the [pre-trainined model](https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin) and put it into the checkpoint direcotry (the file name is `pretrained_h36m_detectron_coco.bin`).

```shell
!wget -c https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin -O checkpoint/pretrained_h36m_detectron_coco.bin
```

## 2. Video Preprocessing

- Prepare a video and put it into video folder.
- Use ffmpeg library to cut the specific part of the video.(the preprocessed video is `cxk.mp4` in the `video` directory)

```shell
!ffmpeg -i video/video1_raw.mp4 -ss 0:11 -to 0:50 -c copy video/video1.mp4
!ffmpeg -i video/video2_raw.mp4 -ss 0:0 -to 0:50 -c copy video/video2.mp4
!ffmpeg -i video/video3_raw.mp4 -ss 0:0 -to 0:25 -c copy video/video3.mp4
!ffmpeg -i video/video4_raw.mp4 -ss 0:0 -to 0:25 -c copy video/video4.mp4
```

## 3. Inferring 2D keypoints with Detectectron

- Set up [Detectron2](https://github.com/facebookresearch/detectron2) and infer 2D keypoints. Prior installing Detectron2, specific version Pytorch and Cuda are installed

```shell
!conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge -y
```

```shell
!python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
```

- Follow the `Inference.md` and make visualizations.

```shell
inference_output_dir = 'inference_output'
video_input_dir = 'video'
!mkdir $inference_output_dir
!cd inference; python infer_video_d2.py \
    --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml \
    --output-dir ../$inference_output_dir \
    --image-ext mp4 \
    ../$video_input_dir
```

## 4: creating a custom dataset

- Run our dataset preprocessing script from the data directory

```shell
video_num = 4
for i in range(video_num):
    input_video = 'video' + str(i+1)
    !cd data; python prepare_data_2d_custom.py -i ../$inference_output_dir -o $input_video
!ls -alt data
```
This creates a custom dataset named myvideos (which contains all the videos in output_directory, each of which is mapped to a different subject) and saved to data_2d_custom_myvideos.npz. You are free to specify any name for the dataset.


Note: as mentioned, the script will take the bounding box with the highest probability for each frame. If a particular frame has no bounding boxes, it is assumed to be a missed detection and the keypoints will be interpolated from neighboring frames.

## 5: rendering a custom video and exporting coordinates
- You can finally use the visualization feature to render a video of the 3D joint predictions. You must specify the custom dataset (-d custom), the input keypoints as exported in the previous step (-k myvideos), the correct architecture/checkpoint, and the action custom (--viz-action custom). The subject is the file name of the input video, and the camera is always 0.

```shell
output_dir = 'video_output'
video_input_dir = 'video'
for i in range(video_num):
    input_video = 'video' + str(i+1)
    input_video_filename = input_video + '.mp4'
    output_video_filename = 'output_video' + str(i+1) + '.mp4'
    !python run.py \
    -d custom \
    -k {input_video} \
    -arc 3,3,3,3,3 \
    -c checkpoint \
    --evaluate pretrained_h36m_detectron_coco.bin \
    --render \
    --viz-subject {input_video_filename} \
    --viz-action custom \
    --viz-video {video_input_dir + "/" + input_video_filename} \
    --viz-output {output_dir + "/" + output_video_filename} \
    --viz-size 6
```

You can also export the 3D joint positions (in camera space) to a NumPy archive. To this end, replace --viz-output with --viz-export and specify the file name.

## Limitations and tips
- The model was trained on Human3.6M cameras (which are relatively undistorted), and the results may be bad if the intrinsic parameters of the cameras of your videos differ much from those of Human3.6M. This may be particularly noticeable with fisheye cameras, which present a high degree of non-linear lens distortion. If the camera parameters are known, consider preprocessing your videos to match those of Human3.6M as closely as possible.
- If you want multi-person tracking, you should implement a bounding box matching strategy. An example would be to use bipartite matching on the bounding box overlap (IoU) between subsequent frames, but there are many other approaches.
- Predictions are relative to the root joint, i.e. the global trajectory is not regressed. If you need it, you may want to use another model to regress it, such as the one we use for semi-supervision.
- Predictions are always in camera space (regardless of whether the trajectory is available). For our visualization script, we simply take a random camera from Human3.6M, which fits decently most videos where the camera viewport is parallel to the ground.