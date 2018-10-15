# mini_project_2 Image Classification
## 1. Tensorflow Method
### 1.1 Prerequisites
#### 1.1.1 TensorFlow install </br>
**Using python pip** </br>
```$ pip install tensorflow```</br>
</br>
**Note**: TensorFlow installed via pip only supports CUDA 9.0. You can build from source following this [tutorial](https://medium.com/@asmello/how-to-install-tensorflow-cuda-9-1-into-ubuntu-18-04-b645e769f01d) if you are using another version of CUDA. You also need to install openJDK before following the instructions</br>
```$ sudo apt-get install openjdk-8-jdk-headless```</br>
</br>
[bazel](https://github.com/bazelbuild/bazel) will be automatically installed as you finished installing openJDK</br>
#### 1.1.2 CUDA install
If you are using GPU for training, download [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) and follow the installing instructions</br>
**IMPORTANT**: You must check your GPU model carefully before installing CUDA. Find the right CUDA version for your GPU. If they do not match, it will not work. </br>
#### 1.1.3 CUDNN install
[CUDNN](https://developer.nvidia.com/cudnn) is required as well. Register a developer's account and download CUDNN. 
#### 1.1.4 openCV install
**Note**: openCV may not work normally if you are using Anaconda. See [this page](https://github.com/ContinuumIO/anaconda-issues/issues/121) for details. </br>
```$ sudo apt-get install python-opencv```
#### 1.1.5 TensorFlow_hub install
```$ pip install tensorflow_hub```
### 1.2 Method Introduction
This method is based on [retraining](https://www.tensorflow.org/hub/tutorials/image_retraining) a pre-trained model (Inception V3 module) trained on [ImageNet](http://image-net.org/). 
### 1.3 Steps
#### Download raw pictures with google-image-download
google-image-download tool can bulk download hundreds of pictures with a single command. See this [link](https://github.com/hardikvasa/google-images-download) for details. 
#### Save raw pictures
Save each class of pictures in separate folders, and name the folders as the class names of the pictures. 
#### Download the retrain source code from tensorflow
Open terminal and go to your desired directory, then download retrain.py</br>
```curl -LO https://github.com/tensorflow/hub/raw/master/examples/image_retraining/retrain.py```
#### Retrain the model
```python retrain.py --image_dir <CLASS_FOLDERS_DIRECTORY>```</br>
</br>
By default, there are 80% training set, 10% validation set and 10% test set. This can be changed using ```--testing_percentage``` and ```--validation_percentage``` commands. 
### 1.4 Results
Run the following code for testing</br>
```curl -LO https://github.com/tensorflow/tensorflow/raw/master/tensorflow/examples/label_image/label_image.py```</br></br>
```python label_image.py --graph=/tmp/output_graph.pb --labels=/tmp/output_labels.txt --input_layer=Placeholder --output_layer=final_result --image=<TEST_IMAGE_PATH>```</br>
<img src="https://github.com/trashcrash/mini_project_2/blob/master/Apple.jpg" width="200"></br>
```
apple 0.99966323
orange 0.00033682885
```
<img src="https://github.com/trashcrash/mini_project_2/blob/master/Orange.jpg" width="200"></br>
```
orange 0.9999039
apple 9.606036e-05
```
<img src="https://github.com/trashcrash/mini_project_2/blob/master/Pear.jpg" width="200"></br>
```
apple 0.903327
orange 0.09667302
```
<img src="https://github.com/trashcrash/mini_project_2/blob/master/Pumpkin.jpg" width="200"></br>
```
apple 0.93159616
orange 0.0684038
```
<img src="https://github.com/trashcrash/mini_project_2/blob/master/Grapefruit.jpg" width="200"></br>
```
orange 0.99999714
apple 2.8362483e-06
```
## 2. Darknet Method (Yolov3)
### 2.1 Prerequisites
#### 2.1.1 Download Yolov3
```git clone https://github.com/pjreddie/darknet```</br>
Pre-trained weights files can be downloaded from the [author's website](https://pjreddie.com/darknet/yolo/). 
#### 2.1.2 Image tagging tools
There are varieties of tagging tools that draw bounding boxes on images. You can choose your own. I used [Yolo_mark](https://github.com/AlexeyAB/Yolo_mark) for my trainings. 
#### 2.1.3 Make config files
For example, make a .data file and put the following lines in it. </br>
```
classes= 2</br>
train  = data/train.txt</br>
valid  = data/test.txt</br>
names = data/obj.names</br>
backup = backup/</br>
```
make a .names file (named as obj.names above), which contains the names of classes. Each line has only one name of the classes. </br>
make train.txt and test.txt</br>
These .txt files contain the paths of training images and validation images. </br>
e.g.</br>
```
data/obj/1.jpg</br>
data/obj/2.jpg</br>
data/obj/3.jpg</br>
data/obj/4.jpg</br>
data/obj/5.jpg</br>
```
After this, you can copy yolov3.cfg file and rename it, and do some modifications following [this tutorial](https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects), like changing the class number and filter number.</br>

#### 2.1.4 Tag the images
There will be txt files generated as you tag the images. Make sure the txt files and corresponding image files have the same name and saved in the same folder. 
#### 2.1.5 Make files
Open Makefile, if you would like to train with CPU, leave the first five values equal to zero. If you would like to train with GPU (which is way faster), change the first three values to 1. Now everything should be set, run ```$ make``` in ```/darknet``` directory. 
### 2.2 Training
You are gonna need a pre-trained weights file as a starting point. Download ```darknet53.conv.74``` from [this site](https://pjreddie.com/darknet/yolo/). </br>
To start training, run something like:</br>
```$ ./darknet detector train data/obj.data cfg/yolo-obj.cfg darknet53.conv.74```</br>
</br>
Newly trained weights are saved every 100 iterations. After 900 iterations, the weights are saved every 10000 iterations, which can be modified by changing line 130 of ```/examples/detector.c```. After the modification is done, go back to ```/darknet``` and run ```$ make clean``` and ```$ make``` again. </br>
The training process can take a long time (days, depending on the number of classes and how much training you do). </br>
### 2.3 Results
I tagged ~1000 door images and ~1500 chair images, and trained for 4600 iterations (about 3 days non-stop). The results are as following. </br>
<img src="https://github.com/trashcrash/mini_project_2/blob/master/chair.jpg" width="400"></br>
```
./test.jpg: Predicted in 0.211778 seconds.
chair: 97%
```
<img src="https://github.com/trashcrash/mini_project_2/blob/master/door.jpg" width="400"></br>
```
./testdoor.jpg: Predicted in 0.214193 seconds.
door: 98%
door: 93%
```
One powerful trait of Yolo is that it can mark items in videos, live or saved locally. </br>
[Here](https://github.com/trashcrash/mini_project_2/blob/master/detection.mp4) is a demo video of Yolo detection using my trained weights (I'm just too lazy to post it here). 
## 3 Compare the two systems
### 3.1 TensorFlow
It could be more powerful than I imagined. I think I just used a tiny part of it. But it worked! Pretty easy to use indeed. There should be some more detailed methods like training with more accurately tagged images. </br>
</br>
It officially supports python wrapper, which means users can use and modify their networks using python, which is good, because most people are too reluctant to use c++ or c. </br>
### 3.2 Darknet-Yolo
Pretty hard to operate, since darknet was written in c. </br>
</br>
Can label each video frame, has huge potential. 
