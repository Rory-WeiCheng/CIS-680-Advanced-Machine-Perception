# CIS 680 Advanced Machine Perception - Project 4: FasterRCNN (Part B) Submission File

## General Description
The submission include 
1. a write-up pdf file that summarize all the required result and our discussion. 
2. The source code of the implementation: `BoxHead Train.ipynb`, `boxHead_Inference.ipynb`, `BoxHead.py`, `create_evaluation.py`, `dataset.py`, `pretrained_models.py`, `utils.py`
3. a "result" filefolder, including the trained boxhead model weight and all the training and validaton loss
4. The README.md instruction (this file)

## BoxHead_Inference.ipynb
The main notebook to run all the code and produce the result. Other than the pdf file, all the required result and visualization are included in the Inference.ipynb notebook as well. **One can reproduce the results through cell evaluation**. Before using, make sure the following packages are available (also see the first cell of the notebook): 

h5py, opencv-python, torch, pytorch_lightning, torchvision, torchsummary, matplotlib, matplotlib, tqdm, numpy

Meanwhile, the data (which is not submitted) should be placed in the "data" filefolder in the same path which includes the pretrained RPN and backbone weight `checkpoint680.pth` and all image files.

In the notebook, **each cell corresponds to each of the requires section of the assignment.**

## BoxHead.py
The python file for boxHead class, which includes the model structure and forward to output classification and regression result. It also has the ground truth creation, post-processing and loss computation function to help train the model and process final results for visualization. For the details and usage of the functions, see the function declaration and comments in the source code.

## dataset.py
The python file for dataset class, it also include the ground truth visualization functions. For the details and usage of the functions, see the function declaration and comments in the source code.

## utils.py
The python file that contain 4 functions that are frequently used in the implementation: MultiApply, IOU_cxcywh, IOU_xyxy, output_decoding. For the details and usage of the functions, see the function declaration and comments in the source code.

## pretrained_models.py
The python file for loading pretrained backbone and RPN model for later boxHead inference. 

## create_evaluation.py
The python file for creating the prediction.npz file for boxHead model inference evaluaiton on hidden image sets. 

## BoxHead_Train.ipynb
The notebook used to train the boxHead model. We have implemented the training process with PyTorch not Pytorch-lightning, which is different from the Part (a) training. This file will not be used unless retrain is needed. We have provided the trained model and loss recordings in the result file folder.