import numpy as np
import tqdm
from losses.dsm import anneal_dsm_score_estimation

import torch.nn.functional as F
import logging
import torch
import os
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from models.ncsnv2 import NCSNv2
from datasets import get_dataset, data_transform, inverse_data_transform
from losses import get_optimizer
from models import (anneal_Langevin_dynamics)
from models import get_sigmas
from models.ema import EMAHelper

__all__ = ['NCSNRunner']

#################################################################
# In this script we define the training and sampling procedures #
#################################################################

# Here we get the model from the NCSNv2 class
def get_model(config):
    if config.data.dataset == 'CIFAR10' or config.data.dataset == 'CELEBA' or config.data.dataset == 'MNIST' or config.data.dataset == 'CAR':
        return NCSNv2(config).to(config.device)

# This is the RUnner class, with all the procedures inside as methods
class NCSNRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config
        args.log_sample_path = os.path.join(args.log_path, 'samples')
        os.makedirs(args.log_sample_path, exist_ok=True)

    # Train procedure
    def train(self):
        # Dataset loading
        dataset, test_dataset = get_dataset(self.args, self.config)
        dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                num_workers=self.config.data.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                 num_workers=self.config.data.num_workers, drop_last=True)
        test_iter = iter(test_loader)
        self.config.input_dim = self.config.data.image_size ** 2 * self.config.data.channels

        tb_logger = self.config.tb_logger

        # We define the score NN as the NCSNv2 model
        score = get_model(self.config)

        score = torch.nn.DataParallel(score)

        # Depending on the configurations, we get the right optimizer chosen
        optimizer = get_optimizer(self.config, score.parameters())

        start_epoch = 0
        step = 0

        # We initialize the EMA (exponential moving average) for slowly upgrading the weights of the NN, ensuring more stability
        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(score)

        # Whether to resume training
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, 'checkpoint.pth'))
            score.load_state_dict(states[0])
            ### Make sure we can resume with different eps
            states[1]['param_groups'][0]['eps'] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        # Get the sigmas from the chosen schedule
        sigmas = get_sigmas(self.config)

        # Training loop
        for epoch in range(start_epoch, self.config.training.n_epochs):
            for i, (X, y) in enumerate(dataloader):
                score.train()
                step += 1

                X = X.to(self.config.device)
                X = data_transform(self.config, X)

                # We compute the dsm loss as defined in the report
                loss = anneal_dsm_score_estimation(score, X, sigmas, None,
                                                   self.config.training.anneal_power)
                tb_logger.add_scalar('loss', loss, global_step=step)

                logging.info("step: {}, loss: {}".format(step, loss.item()))

                # Do an optimizer step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(score)

                if step >= self.config.training.n_iters:
                    return 0

                # every 100 steps we compute the loss on the test dataset
                if step % 100 == 0:
                    if self.config.model.ema:
                        test_score = ema_helper.ema_copy(score)
                    else:
                        test_score = score

                    test_score.eval()
                    try:
                        test_X, test_y = next(test_iter)
                    except StopIteration:
                        test_iter = iter(test_loader)
                        test_X, test_y = next(test_iter)

                    test_X = test_X.to(self.config.device)
                    test_X = data_transform(self.config, test_X)

                    with torch.no_grad():
                        test_dsm_loss = anneal_dsm_score_estimation(test_score, test_X, sigmas, None,
                                                                    self.config.training.anneal_power)
                        tb_logger.add_scalar('test_loss', test_dsm_loss, global_step=step)

                        logging.info("step: {}, test_loss: {}".format(step, test_dsm_loss.item()))

                        del test_score

                # At a chosen frequency we save a chekpoint of the model and sample from it
                if step % self.config.training.snapshot_freq == 0:
                    states = [
                        score.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(states, os.path.join(self.args.log_path, 'checkpoint_{}.pth'.format(step)))
                    torch.save(states, os.path.join(self.args.log_path, 'checkpoint.pth'))

                    # Sampling procedure
                    if self.config.training.snapshot_sampling:
                        if self.config.model.ema:
                            test_score = ema_helper.ema_copy(score)
                        else:
                            test_score = score

                        test_score.eval()

                        init_samples = torch.rand(36, self.config.data.channels,
                                                  self.config.data.image_size, self.config.data.image_size,
                                                  device=self.config.device)
                        init_samples = data_transform(self.config, init_samples)

                        # Start the Annealed Langeving procedure
                        all_samples = anneal_Langevin_dynamics(init_samples, test_score, sigmas.cpu().numpy(),
                                                               self.config.sampling.n_steps_each,
                                                               self.config.sampling.step_lr,
                                                               final_only=True, verbose=True,
                                                               denoise=self.config.sampling.denoise)

                        # We resize and transform the samplesin order to visualize them
                        sample = all_samples[-1].view(all_samples[-1].shape[0], self.config.data.channels,
                                                      self.config.data.image_size,
                                                      self.config.data.image_size)

                        sample = inverse_data_transform(self.config, sample)

                        image_grid = make_grid(sample, 6)
                        save_image(image_grid,
                                   os.path.join(self.args.log_sample_path, 'image_grid_{}.png'.format(step)))
                        torch.save(sample, os.path.join(self.args.log_sample_path, 'samples_{}.pth'.format(step)))

                        del test_score
                        del all_samples

    # Samplig procedure from a checkpoint chosen
    def sample(self):
        if self.config.sampling.ckpt_id is None:
            states = torch.load(os.path.join(self.args.log_path, 'checkpoint.pth'), map_location=self.config.device)
        else:
            states = torch.load(os.path.join(self.args.log_path, f'checkpoint_{self.config.sampling.ckpt_id}.pth'),
                                map_location=self.config.device)

        # Define the Score NN and set the weights loaded from the checkpoint
        score = get_model(self.config)
        score = torch.nn.DataParallel(score)

        score.load_state_dict(states[0], strict=True)
        
        # Load the EMA as well from the checkpoint
        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(score)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(score)

        # Get the chosen schedule
        sigmas_th = get_sigmas(self.config)
        sigmas = sigmas_th.cpu().numpy()

        score.eval()

        # We can sample in order to compute samples as single images for the FID computation or just to produce a grid of images
        if not self.config.sampling.fid:
            init_samples = torch.rand(self.config.sampling.batch_size, self.config.data.channels,
                                        self.config.data.image_size, self.config.data.image_size,
                                        device=self.config.device)
            init_samples = data_transform(self.config, init_samples)

            # Start the Annealed Langeving procedure
            all_samples = anneal_Langevin_dynamics(init_samples, score, sigmas,
                                                    self.config.sampling.n_steps_each,
                                                    self.config.sampling.step_lr, verbose=True,
                                                    final_only=self.config.sampling.final_only,
                                                    denoise=self.config.sampling.denoise)
            
            # We can choose to save only the last sample or all the intermediate steps
            if not self.config.sampling.final_only:
                for i, sample in tqdm.tqdm(enumerate(all_samples), total=len(all_samples),
                                            desc="saving image samples"):
                    sample = sample.view(sample.shape[0], self.config.data.channels,
                                            self.config.data.image_size,
                                            self.config.data.image_size)

                    sample = inverse_data_transform(self.config, sample)

                    image_grid = make_grid(sample, int(np.sqrt(self.config.sampling.batch_size)))
                    save_image(image_grid, os.path.join(self.args.image_folder, 'image_grid_{}.png'.format(i)))
                    torch.save(sample, os.path.join(self.args.image_folder, 'samples_{}.pth'.format(i)))
            else:
                sample = all_samples[-1].view(all_samples[-1].shape[0], self.config.data.channels,
                                                self.config.data.image_size,
                                                self.config.data.image_size)

                sample = inverse_data_transform(self.config, sample)

                image_grid = make_grid(sample, int(np.sqrt(self.config.sampling.batch_size)))
                save_image(image_grid, os.path.join(self.args.image_folder,
                                                    'image_grid_{}.png'.format(self.config.sampling.ckpt_id)))
                torch.save(sample, os.path.join(self.args.image_folder,
                                                'samples_{}.pth'.format(self.config.sampling.ckpt_id)))

        # Here we generate samples with the aim of using them for the FID evaluation, thus they are saved as separated images
        else:
            total_n_samples = self.config.sampling.num_samples4fid
            n_rounds = total_n_samples // self.config.sampling.batch_size

            img_id = 0
            for _ in tqdm.tqdm(range(n_rounds), desc='Generating image samples for FID/inception score evaluation'):
                samples = torch.rand(self.config.sampling.batch_size, self.config.data.channels,
                                        self.config.data.image_size,
                                        self.config.data.image_size, device=self.config.device)
                samples = data_transform(self.config, samples)
                
                # Start the Annealed Langeving procedure
                all_samples = anneal_Langevin_dynamics(samples, score, sigmas,
                                                       self.config.sampling.n_steps_each,
                                                       self.config.sampling.step_lr, verbose=False,
                                                       denoise=self.config.sampling.denoise)

                samples = all_samples[-1]
                for img in samples:
                    img = inverse_data_transform(self.config, img)

                    save_image(img, os.path.join(self.args.image_folder, 'image_{}.png'.format(img_id)))
                    img_id += 1