# TFOD-Setup

Tensorflow Object Detection Setup (MACOS)
Steps are mentioned in following link also (https://c17hawke.github.io/tfod-setup/)


Steps
======
1. Download Tensorflow Repo by clicking Clone->DownloadZip
(https://github.com/tensorflow/models/tree/v1.13.0)
2. Download the model 
(http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz)
3. Download Dataset and Utils
(https://drive.google.com/file/d/12F5oGAuQg7qBM_267TCMt_rlorV-M7gf/view?usp=sharing)
4. Download labelImg tool for labeling images
    1. Clone the Repo (https://github.com/tzutalin/labelImg) or DownloadZip and go inside the folder
    2. Run command "pip install pyqt5==5.13.2 lxml"
    3. Run command "make qt5py3"
    4. Run command "python3 labelImg.py"
5. Unzip all the files 
    1. Raname models-1.13.0.zip -> models
    2. Rename faster_rcnn_inception_v2_coco_2018_01_28 -> faster_rcnn
    3. Inside models folder only keep research folder and delete rest
    
6. Creating virtual env using conda
    1. conda activate tfod
7. Install Packages
    1. pip install pillow lxml Cython contextlib2 jupyter matplotlib pandas opencv-python tensorflow==1.14.0
8. Install protobuf using conda package manager
    1. conda install -c anaconda protobuf
9. For protobuff to .py conversion
    1. Now in the models->research folder run the following command- "protoc object_detection/protos/*.proto --python_out=."

    
