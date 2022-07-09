# Dependencies

- Python 3+ distribution (need to setup on gpu farm)
- [Pytorch](https://pytorch.org/get-started/locally/) >= 0.4.0
  (need to set up on the gpu farm)

# Evaluating the pretrainied model

- Preprocess the Human3.6 DataSet and get npz files(done!, download [data_2d_h36m_cpn_ft_h36m_dbb.npz](https://drive.google.com/file/d/1FfnFpFzoOsJ2kzaY_L9q8buS7t2GClBL/view?usp=sharing) and [data_3d_h36m.npz](https://drive.google.com/file/d/1mFAlUkeIUCguSvYLib21l6VY6ByVFsQk/view?usp=sharing) put it in the `data` folder)
- Download `pretrained_h36m_cpn.bin` and put it into the `checkpoint` folder. (done!)
- Test on Human 3.6m. (done!)

```
python run.py -k cpn_ft_h36m_dbb -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_cpn.bin
```

# Run on custom videos

## 1. Set up

- Use `pip install ffmpeg` to install the specific library. (need to set up on gpu farm)
- Download the [pre-trainined model](https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin) and put it into the checkpoint direcotry.(done, the file name is `pretrained_h36m_detectron_coco.bin`)

## 2. Video Preprocessing

- Prepare a video and put it into video folder.(done!)
- Use ffmpeg library to cut the specific part of the video.(done!, the preprocessed video is `cxk.mp4` in the `video` directory)

```shell
ffmpeg -i demo.mp4 -ss 0:11 -to 0:49 -c copy cxk.mp4
```

## 3. Inferring 2D keypoints with Detectectron

- Set up [Detectron2](https://github.com/facebookresearch/detectron2) and infer 2D keypoints. (need to set up on gpu farm, done!)
- Follow the `Inference.md` and make visualizations (done!).

```shell
cd inference
python infer_video_d2.py \
    --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml \
    --output-dir video_output \
    --image-ext mp4 \
    video
```

## 4: creating a custom dataset

- Run our dataset preprocessing script from the data directory: (done!)

```shell
python prepare_data_2d_custom.py -i /path/to/detections/output_directory -o myvideos
This creates a custom dataset named myvideos (which contains all the videos in output_directory, each of which is mapped to a different subject) and saved to data_2d_custom_myvideos.npz. You are free to specify any name for the dataset.
```

Note: as mentioned, the script will take the bounding box with the highest probability for each frame. If a particular frame has no bounding boxes, it is assumed to be a missed detection and the keypoints will be interpolated from neighboring frames.

## 5: rendering a custom video and exporting coordinates
- You can finally use the visualization feature to render a video of the 3D joint predictions. You must specify the custom dataset (-d custom), the input keypoints as exported in the previous step (-k myvideos), the correct architecture/checkpoint, and the action custom (--viz-action custom). The subject is the file name of the input video, and the camera is always 0. (done!)

```shell
python run.py -d custom -k myvideos -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_detectron_coco.bin --render --viz-subject input_video.mp4 --viz-action custom --viz-camera 0 --viz-video /path/to/input_video.mp4 --viz-output output.mp4 --viz-size 6
```

You can also export the 3D joint positions (in camera space) to a NumPy archive. To this end, replace --viz-output with --viz-export and specify the file name.

## Limitations and tips
- The model was trained on Human3.6M cameras (which are relatively undistorted), and the results may be bad if the intrinsic parameters of the cameras of your videos differ much from those of Human3.6M. This may be particularly noticeable with fisheye cameras, which present a high degree of non-linear lens distortion. If the camera parameters are known, consider preprocessing your videos to match those of Human3.6M as closely as possible.
- If you want multi-person tracking, you should implement a bounding box matching strategy. An example would be to use bipartite matching on the bounding box overlap (IoU) between subsequent frames, but there are many other approaches.
- Predictions are relative to the root joint, i.e. the global trajectory is not regressed. If you need it, you may want to use another model to regress it, such as the one we use for semi-supervision.
- Predictions are always in camera space (regardless of whether the trajectory is available). For our visualization script, we simply take a random camera from Human3.6M, which fits decently most videos where the camera viewport is parallel to the ground.