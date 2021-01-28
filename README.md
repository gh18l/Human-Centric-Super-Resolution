# Human-Centric Super-Resolution based on 3D texture model

<p align="center">
	<img src="https://github.com/li19960612/Human_Model_Render/blob/master/demo/pipeline.PNG" alt="Sample" width="700">
</p>

## Description
The reposity contains the code that generates the human models from high-definition RGB images or videos, binds the detailed texture onto the model, and renders it to low-definition frames according to refined human 3D pose.

Firstly, the project use [SMPL](https://smpl.is.tue.mpg.de/) as basic 3D parameterized model, the [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose), [PSPNet](https://github.com/Lextal/pspnet-pytorch) and the [HMR](https://github.com/akanazawa/hmr) as the supervision of minimizing the error of re-projection of SMPL. Combined with temporal contraint, it can produce a reasonable initial result.

For the mesh reconstruction, the contour of SMPL is expaneded to fit the human silhouette in image to generate new SMPL template. The [Multi-label $\alpha$-expansion](https://vision.cs.uwaterloo.ca/code/) is used to solve a discrete optimition process to find the correspondence between the contours of SMPL and human silhouette. To reduce the peak noise in edge, the laplacian coordination is used to preserve the smoothness of edge.

For the motion refinement, the high frequency details from high-definition motion and low frequency long-term trend from low-definition motion are fused to generate the final results. 

The project use [opendr](https://pypi.org/project/opendr/) as a simple rendering tool. But it may produce a low-quality texture. Alternatively, the [OpenGL](https://www.opengl.org/) is used to generate the final texture. It will upload soon.

## Data Preprocessing
The high and low definition images can be captured with Double Camera System. The best way is to crop the human using detection boxes in each frame. [YOLO](https://github.com/AlexeyAB/darknet) and [KCF](https://github.com/joaofaro/KCFcpp) are recommended to generate the feasible data. 

## Installation
```
conda create -n conda_name python=2.7
pip install -r requirements.txt
pip install pip==8.1.1
pip install opendr==0.78
pip install --upgrade pip
```

## Demo
Because of the low ability to show off high-resolution pictures in GitHub, it is difficult to check the super-resolution results. For convenience, only the results of model reconstrution are shown here.

There are some rendering result in MPII dataset. The
original low-definition human sequence is $\times$ 8 downsampled.

### The results of MPII dataset 
<img src="https://github.com/li19960612/Human_Model_Render/blob/master/demo/LR1.gif" width="400" alt="origin"> <img src="https://github.com/li19960612/Human_Model_Render/blob/master/demo/model1.gif" width="400" alt="model"> 

<img src="https://github.com/li19960612/Human_Model_Render/blob/master/demo/LR2.gif" width="400" alt="origin"> <img src="https://github.com/li19960612/Human_Model_Render/blob/master/demo/model2.gif" width="400" alt="model"> 

And there are also some visual results captured with our double camera system.

### The results in real scene
<img src="https://github.com/li19960612/Human_Model_Render/blob/master/demo/LR3.gif" width="400" alt="origin"> <img src="https://github.com/li19960612/Human_Model_Render/blob/master/demo/model3.gif" width="400" alt="model">

<img src="https://github.com/li19960612/Human_Model_Render/blob/master/demo/LR4.gif" width="400" alt="origin"> <img src="https://github.com/li19960612/Human_Model_Render/blob/master/demo/model4.gif" width="400" alt="model">

## Code
The code is incomplete, the complete implementation will be uploaded soon...






