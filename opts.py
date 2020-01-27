# code from https://github.com/tomguluson92/StyleGAN_PyTorch

import argparse
import torch
import os


def INFO(inputs):
    print("[ Style GAN ] %s" % (inputs))


def presentParameters(args_dict):
    """
        Print the parameters setting line by line

        Arg:    args_dict   - The dict object which is transferred from argparse Namespace object
    """
    INFO("========== Parameters ==========")
    for key in sorted(args_dict.keys()):
        INFO("{:>15} : {}".format(key, args_dict[key]))
    INFO("===============================")


class TrainOptions():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--epoch', type=int, default=500)
        parser.add_argument('--batch_size', type=int, default=2)
        parser.add_argument('--type', type=str, default='style')
        parser.add_argument('--resume', type=str, default='train_result/models/latest.pth')
        parser.add_argument('--det', type=str, default='train_result')
        parser.add_argument('--r1_gamma', type=float, default=10.0)
        parser.add_argument('--r2_gamma', type=float, default=0.0)
        self.opts = parser.parse_args()

    def parse(self):
        self.opts.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Check if the parameter is valid
        if self.opts.type not in ['style', 'origin']:
            raise Exception(
                "Unknown type: {}  You should assign one of them ['style', 'origin']...".format(self.opts.type))

        # Create the destination folder
        if not os.path.exists(self.opts.det):
            os.mkdir(self.opts.det)
        if not os.path.exists(os.path.join(self.opts.det, 'images')):
            os.mkdir(os.path.join(self.opts.det, 'images'))
        if not os.path.exists(os.path.join(self.opts.det, 'models')):
            os.mkdir(os.path.join(self.opts.det, 'models'))

        # Print the options
        presentParameters(vars(self.opts))
        return self.opts


class InferenceOptions():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--resume', type=str, default='train_result/model/latest.pth')
        parser.add_argument('--type', type=str, default='style')
        parser.add_argument('--num_face', type=int, default=32)
        parser.add_argument('--det', type=str, default='result.png')
        self.opts = parser.parse_args()

    def parse(self):
        self.opts.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Print the options
        presentParameters(vars(self.opts))
        return self.opts
