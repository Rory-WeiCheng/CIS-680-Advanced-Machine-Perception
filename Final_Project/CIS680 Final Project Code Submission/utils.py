import numpy as np
import torch
import lpips
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from pytorch_fid.inception import InceptionV3
from scipy import linalg
# plt.switch_backend('agg')


def norm(image):
	'''
	Normalize image tensor
	'''
	return (image/255.0-0.5)*2.0


def denorm(tensor):
	'''
	Denormalize image tensor
	'''
	return ((tensor+1.0)/2.0)*255.0


def reparameterization(z_mean, z_log_var):
	bz, nz = z_mean.shape
	std = torch.exp(z_log_var / 2)
	sampled_z = torch.randn(bz, nz, device=z_mean.device)
	z = sampled_z * std + z_mean
	return z


def save_model_state(generator, encoder, D_VAE, D_LR, e):
	torch.save(generator.state_dict(), f"result/checkpoint/generator-epoch={e+1}.pth")
# 	torch.save(encoder.state_dict(), f"result/checkpoint/encoder-epoch={e+1}.pth")
# 	torch.save(D_VAE.state_dict(), f"result/checkpoint/D_VAE-epoch={e+1}.pth")
# 	torch.save(D_LR.state_dict(), f"result/checkpoint/D_LR-epoch={e+1}.pth")


def loss_plot(loss, title, step=500):
	fig, ax = plt.subplots(1,2,figsize=(12,5))
	smooth_loss = []
	for i in range(0, len(loss), step):
		smooth_loss.append(np.mean(loss[i:i+step]))
	ax[0].plot(loss)
	ax[1].plot(smooth_loss)
	fig.suptitle(title)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def inference_plot(generator, dataset, device, epoch=None, display=True, start_index=0, latent_dim=8, n_sample=10, n_show=5):
    real_edge = []
    real_rgb = []
    infer_result = [[] for _ in range(n_show)]
    with torch.no_grad():
        for idx in range(start_index, start_index+n_sample):
            batch = dataset[idx]
        # for idx, batch in enumerate(dataloader, 0):
        #     # Keep record of n_samples images for inference
        #     if idx == n_sample:
        #         break
            edge_tensor, rgb_tensor = batch
            edge_tensor = edge_tensor.unsqueeze(0)
            rgb_tensor = rgb_tensor.unsqueeze(0)
            real_A = norm(edge_tensor).to(device)
            # Append edge and real rbg
            real_edge.append(edge_tensor)
            real_rgb.append(rgb_tensor)
            # Generate random noise N(z) and feed into generator
            for show_idx in range(n_show):
                N_z = torch.randn(real_A.shape[0], latent_dim, device=device)
                gen_B_random = generator(real_A, N_z)
                fake_denorm = denorm(gen_B_random).cpu()
                infer_result[show_idx].append(fake_denorm)
    # Reshape edge, rgb and infer result
    real_edge = torch.cat(real_edge) # (n_sample,3,128,128)
    real_rgb = torch.cat(real_rgb)   # (n_sample,3,128,128)
    infer_result = [torch.cat(sample_ith) for sample_ith in infer_result] # (n_sample,) {(n_show,3,128,128)}
    # Transform to grid format for visualization
    real_edge_grid = torchvision.utils.make_grid(real_edge, nrow=1).permute(1,2,0).to(torch.uint8)
    real_edge_rgb = torchvision.utils.make_grid(real_rgb, nrow=1).permute(1,2,0).to(torch.uint8)
    # Plot result
    fig, axes = plt.subplots(1, 2+n_show, figsize=(3*(2+n_show), 3*n_sample))
    axes[0].imshow(real_edge_grid)
    axes[0].set_title('Real Edges')
    axes[0].get_xaxis().set_visible(False)
    axes[0].get_yaxis().set_visible(False)
    axes[1].imshow(real_edge_rgb)
    axes[1].set_title('Real RGB')
    axes[1].get_xaxis().set_visible(False)
    axes[1].get_yaxis().set_visible(False)
    for infer_idx in range(n_show):
        infer_ith_grid = torchvision.utils.make_grid(infer_result[infer_idx], nrow=1).permute(1,2,0).to(torch.uint8)
        axes[2+infer_idx].imshow(infer_ith_grid)
        axes[2+infer_idx].get_xaxis().set_visible(False)
        axes[2+infer_idx].get_yaxis().set_visible(False)
    # Save image
    if epoch is not None:
        fig.savefig(f"result/visual_training/inference_plot-epoch={epoch}.png")
    # Display figure if needed
    if not display:
        plt.close(fig)


###################### FID ######################
def build_dataset(dataloader, generator, latent_dim, device):
    gen_set = []
    real_set = []
    with torch.no_grad():
        # Iterate through all validation images to generate fake GAN output
        for idx, batch in enumerate(dataloader, 0):
            edge_tensor, rgb_tensor = batch
            edge_tensor = norm(edge_tensor).to(device)
            real_A = edge_tensor
            # Generate random noise N(z) and feed into generator
            N_z = torch.randn(real_A.shape[0], latent_dim, device=device)
            gen_B_random = generator(real_A, N_z)
            fake_denorm = denorm(gen_B_random)
            gen_set.append(fake_denorm)
            real_set.append(rgb_tensor.to(fake_denorm.device))
        # Transform list of tensors to dataset
        gen_dataset = TensorDataset(torch.cat(gen_set))
        real_dataset = TensorDataset(torch.cat(real_set))
    return real_dataset, gen_dataset


def compute_satistics(model, dataset, device, batch_size=32):
    # Extract feature vector from images with Inception model
    dataloader = DataLoader(dataset, batch_size=batch_size)
    n = len(dataset)
    dim = 2048
    idx_counter = 0
    pred_arr = np.zeros((n, dim))
    for i, batch in enumerate(dataloader, 0):
        # Extract image -> (bz,3,128,128)
        image = batch[0].to(device)
        with torch.no_grad():
            pred = model(image)[0] # (bz, dim, 1, 1)
            pred = pred.squeeze(3).squeeze(2).cpu().numpy() # (bz, dim)
            pred_arr[idx_counter:idx_counter+pred.shape[0]] = pred
            idx_counter += len(pred)
    # Based on activation from Inception model, compute mu and sigma
    mu = np.mean(pred_arr, axis=0)
    sigma = np.cov(pred_arr, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Compute the Frechet Distance between two multivariate Gaussians 
    X_1 ~ N(mu_1, C_1) and X_2 ~ N(mu_2, C_2) as
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2))
    Params:
    mu1:    Numpy array containing the activations of a layer of the
            inception net (like returned by the function 'get_predictions')
            for generated samples.
    mu2:    The sample mean over activations, precalculated on an
            representative data set.
    sigma1: The covariance matrix over activations for generated samples.
    sigma2: The covariance matrix over activations, precalculated on an
            representative data set.
    Returns:
    FID_score: The Frechet Distance.
    """
    # Compute the difference between two distribution's mean
    diff = mu1 - mu2
    # Compute covariance matrix between two distribution
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    # Final expression of FID score
    FID_score = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return FID_score


def FID(dataloader, generator, latent_dim, device):
    # 1. Build real and generated Dataset from given dataloader
    real_dataset, gen_dataset = build_dataset(dataloader, generator, latent_dim, device)
    # 2. load InveptionV3 model
    block_idx = 3
    model = InceptionV3([block_idx]).to(device)
    # 3. Get mean an covariance statistics from Inception model
    m1, s1 = compute_satistics(model, real_dataset, device)
    m2, s2 = compute_satistics(model, gen_dataset, device)
    # 4. Compute FID value
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value
##################################################################


###################### LPIPS ######################
def LPIPS(dataloader, generator, latent_dim, device):
	lpips_score = []
	loss_fn = lpips.LPIPS(net='alex').to(device)
	# Iterate through all validation images to compute LPIPS score
	with torch.no_grad():
		for idx, batch in enumerate(dataloader, 0):
			edge_tensor, _ = batch
			edge_tensor = norm(edge_tensor).to(device)
			real_A = edge_tensor
			# 1. Generate ten B_hat based on real_A input along with random noise
			rand_B_hat = []
			for _ in range(10):
				N_z = torch.randn(real_A.shape[0], latent_dim, device=device)
				gen_B_random = generator(real_A, N_z)
				fake_denorm = denorm(gen_B_random)
				rand_B_hat.append(fake_denorm)
			# 2. Compute pairwise LPIPS distance between those ten samples
			curr_dist = []
			for rand_idx in range(len(rand_B_hat)):
				for rem_idx in range(rand_idx+1, len(rand_B_hat)):
					dist = loss_fn.forward(rand_B_hat[rand_idx],rand_B_hat[rem_idx])
					curr_dist.append(dist.cpu().detach().numpy())
			# 3. Append current image's LPIPS score
			lpips_score.append(np.mean(curr_dist))
	LPIPS_score = np.mean(lpips_score)
	return LPIPS_score
###################################################