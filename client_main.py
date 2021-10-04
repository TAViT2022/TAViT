import os
import argparse
from logging import DEBUG, INFO
from collections import OrderedDict
from itertools import repeat
import timeit

from omegaconf import OmegaConf
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor
import src.client as client
from src.client.tavit_client import TAViTClient
from src.common import (
    parameters_to_weights,
    features_to_weights,
    gradients_to_weights,
    weights_to_parameters,
    weights_to_features,
    weights_to_gradients,
)

from src.common.utils import param_freeze, param_unfreeze
from src.common.logger import log
from src.client.torch_model.dendiffcnn.denoising_diffusion_mysc import Unet_encoder, Unet_decoder
from src.client.torch_model.utils import ReScale, ToYUVTensor, ToRGBTensor, RandomRotate90, img2patch, patch2img
from src.client.data.deblock_dataloader import deblock_dataloader

class MyClient(TAViTClient):
    """TAViT client class for deblocking task. The class contains task-specific head and tail.
    For the detailed usage, check the example of main function.

    :param torch.module head: user defined pytorch module for the task-specific head.
    :param torch.module tail: user defined pytorch module for the task-specific tail.
    :param str device: integer number of gpu to use. None for cpu.
    :param dict train_data: dictionary of dataloader and data length {'loader': dataloader, 'num':int} for train
    :param dict valid_data: dictionary of dataloader and data length {'loader': dataloader, 'num':int} for valid
    :param dict model_config: dictionary that contains configurations of the client model.
    :param dict log_config: dictionary that contains configurations about recording (tensorboard, draw, checkpoint)
    :param dict data_config: dictionary that contains configurations of the data.
    """
    def __init__(self,
                 head,
                 tail,
                 device,
                 train_data,
                 valid_data,
                 model_config,
                 log_config,
                 data_config):
        super().__init__()

        self.task = model_config.params.task
        self.task_specific_epoch = model_config.params.task_specific_epoch

        self.data_config = data_config
        self.prepare_data(train_data, valid_data)

        self.device = device
        self.head = head.to(device)
        self.tail = tail.to(device)

        self.lr = model_config.params.lr
        self.evaluate_phase = False

        self.configure_optimizers()
        self.loss = nn.MSELoss()
        self.loss_history = []

        self.writer = SummaryWriter(log_dir=log_config.log_save_dir)

        self.model_save_dir = log_config.log_save_dir
        self.draw_save_dir = log_config.log_save_dir + '/samples/'
        self.log_save_period = log_config.log_save_period

        self.current_epoch = 1

    def move_to_device(self, device):
        self.head = self.head.to(device)
        self.tail = self.tail.to(device)

    def configure_optimizers(self):
        # define the same optimizer for the head and the tail parameters.
        self.optim = torch.optim.Adam(list(self.head.parameters())+list(self.tail.parameters()), lr=self.lr,
                                      betas=[0.5, 0.999])

    def prepare_data(self, train_data, valid_data): 
        self.train_loader = train_data['loader']
        self.valid_loader = valid_data['loader']
        self.train_data_num = train_data['num']
        self.val_data_num = valid_data['num']

        self.reset_dataloader_iter()

    def reset_dataloader_iter(self):
        try:
            del self.train_loader_iter
        except AttributeError:
            pass
        
        self.train_loader_iter = iter(self.train_loader)

    def set_current_data(self):
        try:
            if not self.evaluate_phase:
                x, y, _ = next(self.train_loader_iter)

        except StopIteration:
            # When 'step_per_epoch' of server configuration is larger than the number of dataset,
            self.reset_dataloader_iter()
            if not self.evaluate_phase:
                x, y, _ = next(self.train_loader_iter)

        self.current_input = x.to(self.device)
        self.current_label = y.to(self.device)

    def get_initstates(self):
        # return the initial states of self to the server.
        head_param_num, tail_param_num = self.get_num_parameters()
        return self.task, self.task_specific_epoch, self.train_data_num, self.val_data_num, head_param_num, tail_param_num

    def get_parameters(self):
        # return the paraeters of head and tail to the server.
        head_params = [val.cpu().numpy() for _, val in self.head.state_dict().items()]
        tail_params = [val.cpu().numpy() for _, val in self.tail.state_dict().items()]

        # Because the number of parameters was sent to the server as initstate,
        # we don't need to send these twice.
        params = head_params + tail_params

        return params
    
    def get_num_parameters(self):
        head_params = [val.cpu().numpy() for _, val in self.head.state_dict().items()]
        tail_params = [val.cpu().numpy() for _, val in self.tail.state_dict().items()]
        return len(head_params), len(tail_params)

    def set_parameters(self, parameters):
        # This function is used for 
        # 1. client model initialize using global params
        # 2. weights unifying after weight aggregation at the server.
        head_param_num = len(self.head.state_dict().keys())

        head_parameters = parameters[:head_param_num]
        tail_parameters = parameters[head_param_num:]

        # head parameters
        head_params_dict = zip(self.head.state_dict().keys(), head_parameters)
        head_state_dict = {}

        for k, v in head_params_dict:
            if 'num_batches_tracked' in k:
                head_state_dict[k] = torch.Tensor([v[()]])
            else:
                head_state_dict[k] = torch.Tensor(v)

        head_state_dict = OrderedDict(head_state_dict)
        self.head.load_state_dict(head_state_dict, strict=True)

        # tail parameters
        tail_params_dict = zip(self.tail.state_dict().keys(), tail_parameters)
        tail_state_dict = {}

        for k, v in tail_params_dict:
            if 'num_batches_tracked' in k:
                tail_state_dict[k] = torch.Tensor([v[()]])
            else:
                tail_state_dict[k] = torch.Tensor(v)

        tail_state_dict = OrderedDict(tail_state_dict)
        self.tail.load_state_dict(tail_state_dict, strict=True)

    def forward(self, parameters, config):
        # if the iteration is not unifying iter, parameters will be None.
        # In this case, we do not set and aggregate the client's weights in the server
        if parameters is not None and parameters != []:
            self.set_parameters(parameters)

        if config['evaluate']:
            self.evaluate_phase = True
        else:
            self.evaluate_phase = False

        self.set_current_data()

        # One can change this line depending on the head model, but should keep 'head_output'.
        self.head_output, self.t, self.h = self.head(self.current_input)

        return [self.head_output.clone().cpu().detach().numpy()]
    
    def backward(self, features, config):
        # init gradients of the tail
        features = torch.tensor(features, dtype=torch.float32, requires_grad=True)
        # add skip connection
        output = self.tail(features.to(self.device), self.t, self.h)
        
        self.optim.zero_grad()

        loss = self.loss(input=output, target=self.current_label)

        if config['body_update']:
            loss.backward()
            # When body is updated, client's tail does not receive anything.
            # Thus, we should delete head output and time which are saved to self
            # to avoid unused the GPU memory allocation.
            # https://discuss.pytorch.org/t/gpu-memory-consumption-increases-while-training/2770/2
            del self.head_output, self.t, self.h
        else:
            # We will call backward() method twice. Once here, and another in self.update().
            # Thus, we need to set retain_graph to True, for the second backward().
            loss.backward(retain_graph=True)

        grad = features.grad.clone().cpu().numpy()

        with torch.no_grad():
            # record the average loss of each epoch to tensorboard
            self.loss_history.append(loss.item())
            if config['round'] != self.current_epoch:
                loss_avg = sum(self.loss_history)/len(self.loss_history)
                self.writer.add_scalar('Loss/train', loss_avg, self.current_epoch)
                self.writer.flush()
                self.loss_history = []
                self.current_epoch = config['round']

            # Save head / tail model checkpoint as statedict.
            # Also, draw sample images.
            if config['round'] % self.log_save_period == 0:
                head_save_path = self.model_save_dir + 'head_state_dict_{}.pt'.format(config['round'])
                tail_save_path = self.model_save_dir + 'tail_state_dict_{}.pt'.format(config['round'])

                if not os.path.isfile(head_save_path):
                    torch.save(self.head.state_dict(), head_save_path)
                if not os.path.isfile(tail_save_path):
                    torch.save(self.tail.state_dict(), tail_save_path)

                self.draw_output(output, config['round'])

            if config['reset_data']:
                self.reset_dataloader_iter()

        return [grad]

    def update(self, gradients, config):
        # Get the gradients and convert to tensor.
        gradients = torch.tensor(gradients).to(self.device)

        # Apply the gradient to the head output.
        self.head_output.backward(gradient=gradients)

        # do update head and tail
        self.optim.step()

        if config['get_params']:
            params = self.get_parameters()
        else:
            params = [None]

        return params, 1, {}

    def draw_output(self, output:torch.tensor, round:int):
        # output shape = (bs, 3, 128, 128)

        def norm_scale(x:torch.tensor, mini:int=0, maxi:int=1):
            return x/2+0.5

        in_img, out_img, label_img = map(lambda x: torch.tensor(x.data.cpu().detach().numpy()),\
            [self.current_input, output, self.current_label])

        sample_indices = [0,1,2,3,4]
        grids = []
        for idx in sample_indices:
            grids.append(norm_scale(in_img[idx,:,:,:]))
            grids.append(norm_scale(out_img[idx,:,:,:]))
            grids.append(norm_scale(label_img[idx,:,:,:]))

        os.makedirs(self.draw_save_dir, exist_ok=True)
        torchvision.utils.save_image(grids, os.path.join(self.draw_save_dir, 'Trainlog_img_{}.png'.format(round)), nrow=3)


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--server_address', type=str, default='localhost:8080')
    parser.add_argument('--config_path', type=str, default='src/client/config/client_config.yaml')
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()

    DEVICE = torch.device("cuda:{}".format(args.gpu) if (torch.cuda.is_available() and args.gpu != 'None') else "cpu")
    config = OmegaConf.load(args.config_path)
    
    # Define task-specific dataloader. You need to prepare dict{'loader':dataloader, 'num':int}.
    train_data = deblock_dataloader(**config.data_config, Q=[10, 50], valid=False)
    valid_data = deblock_dataloader(**config.data_config, Q=10, valid=True)

    # Define a task-specific head and tail. 
    head_model = Unet_encoder(input_nc=3, dim=128, dim_mults=(1, 2, 4))
    tail_model = Unet_decoder(input_nc=3, dim=128, dim_mults = (1, 2, 4), skip='add')

    # Client instantiate.
    my_client = MyClient(head=head_model,
                         tail=tail_model,
                         device=DEVICE,
                         train_data = train_data,
                         valid_data = valid_data,
                         **config)

    # Run client. Connect to server (insecure connection)
    client.app.start_tavit_client(args.server_address, client=my_client)

if __name__ == '__main__':
    main()
