# Real-time Object Detection

Simple implementation of an object detector in Python. Screen recording is implemented with OpenCV and the frames are converted to a PIL image to be processed by the [facebook/detr-resnet-50](https://huggingface.co/facebook/detr-resnet-50) model (End-to-End Object Detection model with ResNet-50 backbone). The model's outputs are unpacked and a camera image is created with a rectangle and the name indicating each object.

## Running
To run this object detection, save this dir and run `cd path/to/dir`, then install the dependencies shown in the environment.yml file, or, if you have [anaconda](https://www.anaconda.com), run:
```
conda env create -f environment.yml
conda activate object_detection_env
python.exe detr.py
```

And your detector is running. Press the 'q' key to get out.

It's important to remember that for the code to perform well, you need a GPU.

Check the available cameras: `python.exe camera_check.py` (8 camera indexes will be checked).

