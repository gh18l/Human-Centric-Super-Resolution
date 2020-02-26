# Human_Model_Render

<p align="center">
	<img src="https://github.com/li19960612/Human_Model_Render/blob/master/demo/pipeline.PNG" alt="Sample" width="700">
</p>

## Description
The reposity contains the code that generates the human models from rgb images or videos, binds the human texture onto the model, and renders it to screen. The texture can be arbitrary. The code use the high-definition version of input images as the texture, it looks like reference-based human super resolution.

For the motion, the project use [SMPL](https://smpl.is.tue.mpg.de/) as basic model, the [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose), [PSPNet](https://github.com/Lextal/pspnet-pytorch) and the [HMR](https://github.com/akanazawa/hmr) as the constraint of re-projection. Combined with reasonable temporal contraint, it can produce a reasonable result. 

For the mesh reconstruction, the project expand the contour of SMPL to fit the human silhouette in image to generate new template. The [Multi-label $\alpha$-expansion](https://vision.cs.uwaterloo.ca/code/) is used to find the correspondence between the contours of SMPL and image. To reduce the peak noise in edge, the laplacian coordination is used to preserve the smoothness of edge.

The project use [opendr](https://pypi.org/project/opendr/) as a simple rendering tool. But it may produce a low-quality texture. Alternatively, the [OpenGL](https://www.opengl.org/) is used to generate the final texture. 

## Data Preprocessing
The high and low definition images can be captured with common multi-camera surveillance, such as Hikvision. The best way is to crop the human using detection. [YOLO](https://github.com/AlexeyAB/darknet) and [KCF](https://github.com/joaofaro/KCFcpp) are recommended to generate the feasible data. 

## Installation
```
conda create -n conda_name python=2.7
pip install -r requirements.txt
pip install pip==8.1.1
pip install opendr==0.78
pip install --upgrade pip
```

## Demo
There are some rendering result in MPII dataset. The
original low-definition human sequence is $\times$ 8 downsampled. The sequence with final texture is not demonstrated because of the size-limited in github.

### The results of MPII dataset 
<img src="https://github.com/li19960612/Human_Model_Render/blob/master/demo/LR1.gif" width="400" alt="origin"> <img src="https://github.com/li19960612/Human_Model_Render/blob/master/demo/model1.gif" width="400" alt="model"> 

<img src="https://github.com/li19960612/Human_Model_Render/blob/master/demo/LR2.gif" width="400" alt="origin"> <img src="https://github.com/li19960612/Human_Model_Render/blob/master/demo/model2.gif" width="400" alt="model"> 

And There are also some visual results captured with our hybrid camera system similar to Hikvision surveillance.

### The results in real scene
<img src="https://github.com/li19960612/Human_Model_Render/blob/master/demo/LR3.gif" width="400" alt="origin"> <img src="https://github.com/li19960612/Human_Model_Render/blob/master/demo/model3.gif" width="400" alt="model">





