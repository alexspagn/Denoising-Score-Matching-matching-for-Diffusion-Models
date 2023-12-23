import torch
import numpy as np

############################################################################################
# In this script we define the loss schedules and the Annealed Langevin Dynamics algorithm #
############################################################################################

# Here we define all the noise schedules that we will use
def sigmoid(x):
    return -1 / (1 + np.exp(-x))

def get_sigmas(config):
    if config.model.sigma_dist == 'geometric':
        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(config.model.sigma_begin), np.log(config.model.sigma_end),
                               config.model.num_classes))).float().to(config.device)
    elif config.model.sigma_dist == 'uniform':
        sigmas = torch.tensor(np.linspace(config.model.sigma_begin, config.model.sigma_end, config.model.num_classes)).float().to(config.device)
    elif config.model.sigma_dist == 'linear':
        sigmas = torch.tensor(
                            np.linspace(config.model.sigma_begin, config.model.sigma_end,
                                               config.model.num_classes)).float().to(config.device)
    elif config.model.sigma_dist == 'cosine':
        sigmas = torch.tensor(
                            config.model.sigma_begin * np.cos((np.linspace(0, 1,
                                               config.model.num_classes) / 1.0064) * np.pi / 2)).float().to(config.device)
    elif config.model.sigma_dist == 'cosine_squared':
        sigmas = torch.tensor(
                            config.model.sigma_begin * np.cos((np.linspace(0, 1,
                                               config.model.num_classes) / 1.068) * np.pi / 2) ** 2).float().to(config.device)
    elif config.model.sigma_dist == 'sigmoid':
        sigmas = torch.tensor(
                 (sigmoid(np.linspace(0, 10, config.model.num_classes) - 2.5) + 1.0013)).float()
        sigmas = sigmas - 0.09012*sigmas.max()/config.model.sigma_begin
        sigmas = config.model.sigma_begin * sigmas / sigmas.max()
        sigmas = sigmas.to(config.device)
    else:
        raise NotImplementedError('sigma distribution not supported')

    return sigmas

# Here we define the Annelaed Langevin Dynamics algorithm (Algorithm 1 in the Report)
@torch.no_grad()
def anneal_Langevin_dynamics(x_mod, scorenet, sigmas, n_steps_each=200, step_lr=0.000008,
                             final_only=False, verbose=False, denoise=True):
    images = []

    with torch.no_grad():
        # For every sigma, for a given number of steps for each one, we compute the score and use it in the Langevin equation
        for c, sigma in enumerate(sigmas):
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2
            for s in range(n_steps_each):
                grad = scorenet(x_mod, labels)

                noise = torch.randn_like(x_mod)
                grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
                x_mod = x_mod + step_size * grad + noise * np.sqrt(step_size * 2)

                image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()
                snr = np.sqrt(step_size / 2.) * grad_norm / noise_norm
                grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2

                if not final_only:
                    images.append(x_mod.to('cpu'))
                if verbose:
                    print("level: {}, step_size: {}, grad_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
                        c, step_size, grad_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))

        # Performs a last step of the Langevin without the gaussian noise
        if denoise:
            last_noise = (len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)
            last_noise = last_noise.long()
            x_mod = x_mod + sigmas[-1] ** 2 * scorenet(x_mod, last_noise)
            images.append(x_mod.to('cpu'))

        if final_only:
            return [x_mod.to('cpu')]
        else:
            return images