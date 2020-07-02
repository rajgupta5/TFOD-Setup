# TFOD-Setup

Tensorflow Object Detection Setup (MACOS)

Steps are mentioned in following link also (https://c17hawke.github.io/tfod-setup/)
youtube video (https://www.youtube.com/watch?v=N8lWJmy2_jk)

TFOD API Tutorial
https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/index.html

Tensorflow Model Garden
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

COCO Dataset
https://cocodataset.org/#explore

Object Detection API Demo
https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/object_detection_tutorial.ipynb


TFOD Tutorial
https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10


Steps
======

## Downloading

0. Create a folder "TFOD" at some location and go inside the folder, follow below steps
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
    4. Inside utils we have "images->test/train, training->labelmap.pbtxt, generate_tfrecord.py, xml_to_csv.py"


## Installation

6. Creating virtual env using conda
    1. conda create -n tfod python=3.6
    2. conda activate tfod
7. Install Packages
    1. pip install pillow lxml Cython contextlib2 jupyter matplotlib pandas opencv-python tensorflow==1.14.0
8. Install protobuf using conda package manager
    1. conda install -c anaconda protobuf
9. For protobuff to .py conversion
    1. Now in the TFOD->models->research folder run the following command- "protoc object_detection/protos/*.proto --python_out=."
10. Go Inside TFOD->models->research folder and run command "python setup.py install" for installing object detection API


## Setup Verification 

11. Verification step to see if everything so far is working fine.
    1. jupyter notebook
    2. open file "object_detection_tutorial.ipynb" from TFOD->models->research->object_detection folder
    3. Run all the cells
    4. in last cell run below:
            %matplotlib inline
            plt.figure(figsize=(50,50))
            plt.imshow(image_np)


## Custom Training

12. Paste all content present in utils into research folder. 
    Inside utils we have "images->test/train, training->labelmap.pbtxt, generate_tfrecord.py, xml_to_csv.py"
13. Paste the downloaded model "faster_rcnn" into research folder
    1. Cut and Paste "faster_rcnn" folder inside models->research folder
    2. cd to the research folder and run the following python command "python xml_to_csv.py"
14. Run the following to generate train and test records from research folder
    1. python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record
    2. python generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=test.record
15. Copy from research/object_detection/samples/config/ faster_rcnn_inception_v2_coco.config file into research/training
16. In the config file, Update 
        1. num_classes=6, (no of classes can be found out in labelmap.pbtxt inside research->training folder)
        2. fine_tune_checkpoint = "faster_rcnn/model.ckpt", 
        3. num_steps = 250
        4. input_path for both train_input_reader -> train.record and eval_input_reader -> test.record
        5. label_map_path for both train_input_reader and eval_input_reader -> training/labelmap.pbtxt
17. From research/object_detection/legacy/ copy train.py to research folder
18. Copy deployment and nets folder from research/slim into the research folder
19. NOW Run the following command from the research folder. This will start the training in your local system
    1. python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_coco.config





    
