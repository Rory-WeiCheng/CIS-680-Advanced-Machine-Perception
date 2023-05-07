# CIS 680 Project 4 FasterRCNN Part A: Region Proposal Network (RPN) Submission File
## General Description
The submission include 
1. a write-up pdf file that summarize all the required result and our discussion. 
2. The source code of the implementation: Inference.ipynb, dataset.py, utils.py, rpn.py, RPN_train.ipynb
3. a "result" filefolder, including the trained model and all the training and validaton loss
4. The README.md instruction (this file)
## Inference.ipynb
The main notebook to run all the code and produce the result. Other than the pdf file, all the required result and visualization are included in the Inference.ipynb notebook as well. **One can reproduce the results through cell evaluation**. Before using, make sure the following packages are available (also see the first cell of the notebook): 

h5py, opencv-python, torch, pytorch_lightning, torchvision, torchsummary, matplotlib

Meanwhile, the data(which is not submitted) should be placed in the "data" filefolder in the same path.

In the notebook, **each cell corresponds to each of the requires section of the assignment.**

## rpn.py
The python file for Region Proposal Network class, it also includes the FPN result visualization functions and training steps definition. For the details and usage of the functions, see the function declaration and comments in the source code

## dataset.py
The python file for dataset class, it also include the ground truth visualization functions. For the details and usage of the functions, see the function declaration and comments in the source code 

## utils.py
contain 4 functions that are frequently used in the implementation: MultiApply, calc_IOU, output_flattening, output_decoding.
For the details and usage of the functions, see the function declaration and comments in the source code

## RPN_train.ipynb
The notebook used to call the training of the RPN. This file will not be used unless retrain is needed. We have provided the trained model and loss recordings in the result file folder.
