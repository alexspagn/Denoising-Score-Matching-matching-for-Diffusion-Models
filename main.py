import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import torch.utils.tensorboard as tb
import copy
from runners import *

import os

#############################################################################################################
# This is the main script, from which we can choose which procedure (training or sampling) we want to start #
# and which configuration file we want to choose.                                                           #
#############################################################################################################

# Training example:
# python main.py --config car.yml --doc Car_geom
#
# or, to resume a training:
# python main.py --config car.yml --doc Car_geom --resume_training

# Sampling example:
# python main.py --sample --config CAR.yml --doc Car_sigmoid -i Car_sigmoid_samples

# Function to parse the arguments
def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--config', type=str, required=True,  help='Path to the config file')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--exp', type=str, default='exp', help='Path for saving running related data.')
    parser.add_argument('--doc', type=str, required=True, help='A string for documentation purpose. '
                                                               'Will be the name of the log folder.')
    parser.add_argument('--comment', type=str, default='', help='A string for experiment comment')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('--sample', action='store_true', help='Whether to produce samples from the model')
    parser.add_argument('--resume_training', action='store_true', help='Whether to resume training')
    parser.add_argument('-i', '--image_folder', type=str, default='images', help="The folder name of samples")

    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, 'logs', args.doc)

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    new_config = dict2namespace(config)

    tb_path = os.path.join(args.exp, 'tensorboard', args.doc)

    # This code differentiate between sampling and training procedure, allowing also to resume the training
    if not args.sample:
        if not args.resume_training:
            if os.path.exists(args.log_path):
                overwrite = False
                response = input("Folder already exists. Overwrite? (Y/N)")
                if response.upper() == 'Y':
                    overwrite = True

                if overwrite:
                    shutil.rmtree(args.log_path)
                    shutil.rmtree(tb_path)
                    os.makedirs(args.log_path)
                    if os.path.exists(tb_path):
                        shutil.rmtree(tb_path)
                else:
                    print("Folder exists. Program halted.")
                    sys.exit(0)
            else:
                os.makedirs(args.log_path)

            with open(os.path.join(args.log_path, 'config.yml'), 'w') as f:
                yaml.dump(new_config, f, default_flow_style=False)

        new_config.tb_logger = tb.SummaryWriter(log_dir=tb_path)
        # setup logger
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError('level {} not supported'.format(args.verbose))

        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args.log_path, 'stdout.txt'))
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

    else:
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError('level {} not supported'.format(args.verbose))

        handler1 = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler1.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.setLevel(level)

        os.makedirs(os.path.join(args.exp, 'image_samples'), exist_ok=True)
        args.image_folder = os.path.join(args.exp, 'image_samples', args.image_folder)
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            overwrite = False
            response = input("Image folder already exists. Overwrite? (Y/N)")
            if response.upper() == 'Y':
                overwrite = True

            if overwrite:
                shutil.rmtree(args.image_folder)
                os.makedirs(args.image_folder)
            else:
                print("Output image folder exists. Program halted.")
                sys.exit(0)

    # Add device
    device = torch.device('cuda:0') #if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config

# These function is a utils to convert from dict entries to namespace object
def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))
    logging.info("Config =")
    print(">" * 80)
    config_dict = copy.copy(vars(config))
    if not args.sample:
        del config_dict['tb_logger']
    print(yaml.dump(config_dict, default_flow_style=False))
    print("<" * 80)

    # Start the Runner with the right procedure
    try:
        runner = NCSNRunner(args, config)
        if args.sample:
            runner.sample()
        else:
            runner.train()
    except:
        logging.error(traceback.format_exc())

    return 0


if __name__ == '__main__':
    sys.exit(main())
