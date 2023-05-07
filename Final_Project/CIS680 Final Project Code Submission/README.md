# CIS 680 Advanced Machine Perception - Final Project Instruction

## **Set-up**
Before doing the inference step, first make sure the edge2shoes dataset images are correctly stored under `data/edges2shoes` directory. Since we are using customized dataset storage method, the raw images are first read in and then saved as tensor for later ease of use. To do that, first open the `Train_BicycleGAN.ipynb` file and go to the 
*Generate customized dataset* section and run the cell to generate the training and validation dataset. 

## **Inference**
To inference with the model prediction result, first go to the *Train* section and change the `img_dir_train` and `img_dir_val` dataset path to be the exact location of the generated customized dataset. Then, run the first cell under *Train* section to load in the training and validation dataset.

After those steps, you can scroll down to Inference plot section and run the cell to generate the inference result from the specified dataset. If you want to try different model weight, you can change the model weight path accordingly. 

NOTE: if you want to try ResNet based generator model then you have to change the generator definition from `generator = Generator().to(device)` to `generator = GeneratorResNet(input_shape=img_shape,latent_dim=latent_dim).to(device)` to load the ResNeet based generator model architecture.

## **File explanation**
## models.py
The python file for discriminator, encoder and generator class, which includes the model structure and forward operation. For the generator, there are U-Net and ResNet block to produce different generator backbone structure.

## dataset_preload.py
The python file for read in the customized dataset. It is adapted based on the given dataset.py file with similar structure.

## utils.py
The python file that contains a list of helper functions to run the training and inference step. It mainly consists the plot function, norm & denorm function, inference plot function, FID score calculation function and LPIPS metric calculation function. 