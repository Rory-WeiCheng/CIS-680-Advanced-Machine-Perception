import pdb
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from utils import *


##############################
#           Encoder          #
##############################
""" 
The encoder used in both cVAE-GAN and cLR-GAN, which encode image B or B_hat to latent vector
This encoder uses resnet-18 to extract features, and further encode them into a distribution
similar to VAE encoder. 
Args in constructor: 
    latent_dim: latent dimension for z 
Args in forward function: 
    img: image input (from domain B)       
Returns: 
    mu: mean of the latent code 
    logvar: sigma of the latent code 
"""
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        # Extracts features at the last fully-connected
        resnet18_model = resnet18(pretrained=True)      
        self.feature_extractor = nn.Sequential(*list(resnet18_model.children())[:-3])
        self.pooling = nn.AvgPool2d(kernel_size=8, stride=8, padding=0)
        # Output is mu and log(var) for reparameterization trick used in VAEs
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, img):
        out = self.feature_extractor(img) # (bz,3,128,128) -> (bz,256,8,8)
        out = self.pooling(out)           # (bz,256,8,8) -> (bz,256,1,1)
        out = out.view(out.size(0), -1)   # (bz,256,1,1) -> (bz,256)
        mu = self.fc_mu(out)              # (bz, 8)
        logvar = self.fc_logvar(out)      # (bz, 8)
        return mu, logvar


##############################
#        Discriminator       #
##############################
""" 
The discriminator used in both cVAE-GAN and cLR-GAN
Args in constructor: 
    in_channels: number of channel in image (default: 3 for RGB)
Args in forward function: 
    x: image input (real_B, fake_B)
Returns: 
    discriminator output: could be a single value or a matrix depending on the type of GAN
"""
class Discriminator(nn.Module):

    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        # Number of channels
        self.nc = in_channels
        # Define sub-block of discriminator model
        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, True))
            return layers
        # Define PatchGAN discriminator model
        self.model = nn.Sequential(
            *discriminator_block(self.nc, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, x):
        outputs = self.model(x)
        return outputs

    def compute_loss(self, disc_pred_real, disc_pred_fake, mse_loss):
        '''
        Compute the MSE between discriminator prediction and ground truth label
        '''
        D_real = mse_loss(disc_pred_real, torch.ones_like(disc_pred_real))
        D_fake = mse_loss(disc_pred_fake, torch.zeros_like(disc_pred_fake))
        D_total = D_real + D_fake
        return D_total



##############################
#          Generator         #
##############################
'''
The generator used in both cVAE-GAN and cLR-GAN, which transform A to B
Args in constructor: 
    latent_dim: latent dimension for z
    image_shape: (channel, h, w), you may need this to specify the output dimension (optional)
Args in forward function: 
    x: image input (from domain A)
    z: latent vector (encoded B)
Returns: 
    fake_B: generated image in domain B
'''
class Generator(nn.Module):
    def __init__(self, nz=8, num_downs=7, ngf=64):
        super(Generator, self).__init__()
        input_nc = 3
        output_nc = input_nc
        # construct unet structure
        unet_block = UnetBlock(ngf * 8, ngf * 8, ngf * 8, nz, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetBlock(ngf*8, ngf*8, ngf*8, nz, unet_block)
        unet_block = UnetBlock(ngf*4, ngf*4, ngf*8, nz, unet_block)
        unet_block = UnetBlock(ngf*2, ngf*2, ngf*4, nz, unet_block)
        unet_block = UnetBlock(ngf, ngf, ngf*2, nz, unet_block)
        unet_block = UnetBlock(input_nc, output_nc, ngf, nz, unet_block, outermost=True)

        self.model = unet_block

    def forward(self, x, z):
        return self.model(x,z)


##############################
#          UnetBlock         #
##############################
class UnetBlock(nn.Module):
    def __init__(self, input_nc, outer_nc, inner_nc, nz,
                 submodule=None, outermost=False, innermost=False):
        super(UnetBlock, self).__init__()
        # Parameter 
        self.outermost = outermost
        self.innermost = innermost
        input_nc = input_nc + nz
        # Construct down and up block parts
        downconv = [nn.ReflectionPad2d(1)]
        downconv += [nn.Conv2d(input_nc, inner_nc,
                               kernel_size=4, stride=2, padding=0)]
        downrelu = nn.LeakyReLU(0.2, True)
        uprelu = nn.ReLU()
        # Construct whole block based on layer type
        if outermost:
            upconv = upsampleLayer(inner_nc * 2, outer_nc)
            down = downconv + [norm_layer(inner_nc)]
            up = [uprelu] + upconv + [nn.Tanh()]
        elif innermost:
            upconv = upsampleLayer(inner_nc, outer_nc)
            down = [downrelu] + downconv
            up = [uprelu] + upconv + [norm_layer(outer_nc)]
        else:
            upconv = upsampleLayer(inner_nc * 2, outer_nc)
            down = [downrelu] + downconv + [norm_layer(inner_nc)]
            up = [uprelu] + upconv + [norm_layer(outer_nc)]
        # Construct down, submodule and up block of U-Net
        self.down = nn.Sequential(*down)
        self.submodule = submodule
        self.up = nn.Sequential(*up)


    def forward(self, x, z):
        # Concatenate x with z
        z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
        x_and_z = torch.cat([x, z_img], 1)
        # Forward the concatenated input through different block
        if self.outermost:  
            x1 = self.down(x_and_z)
            x2 = self.submodule(x1, z)
            return self.up(x2)
        elif self.innermost:
            x1 = self.up(self.down(x_and_z))
            return torch.cat([x1, x], 1)
        else:
            x1 = self.down(x_and_z)
            x2 = self.submodule(x1, z)
            return torch.cat([self.up(x2), x], 1)


def upsampleLayer(inplanes, outplanes):
    upconv = [nn.Upsample(scale_factor=2),
            #   nn.ReflectionPad2d(1),
              nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)]
    return upconv

def norm_layer(layer_nc):
    return nn.InstanceNorm2d(layer_nc, affine=False, track_running_stats=False)



#################################
####### ResNet Generator ########
#################################
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks=6, latent_dim=8):
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels + latent_dim, out_features, 7),
            # nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x, z):
        z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
        x_and_z = torch.cat([x, z_img], 1)
        return self.model(x_and_z)
