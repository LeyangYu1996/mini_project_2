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
#### Apple
<img src="https://github.com/trashcrash/mini_project_2/blob/master/Apple.jpg" width="200"></br>
apple 0.99966323</br>
orange 0.00033682885</br>
<img src="https://github.com/trashcrash/mini_project_2/blob/master/Orange.jpg" width="200"></br>
orange 0.9999039</br>
apple 9.606036e-05</br>
