import argparse

import torch
from omegaconf import OmegaConf
import src.server.app as app
from src.server.client_manager import TAViTClientManager
from src.server.tavit_server import TAViTServer
from src.server.strategy import CustomStrategy, FedAvg, Strategy
from src.server.torch_model.transformer.transformer import cnnTransformer

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config_path', type=str, default='src/server/config/server_config.yaml')
    parser.add_argument('--server_address', type=str, default='localhost:8080')
    parser.add_argument('--gpu', type=str, default=0)
    args = parser.parse_args()

    DEVICE = torch.device("cuda:{}".format(args.gpu) if (torch.cuda.is_available() and args.gpu != 'None') else "cpu")
    config = OmegaConf.load(args.config_path)

    client_manager = TAViTClientManager()
    strategy = CustomStrategy(min_available_clients=config.server_config.num_clients,
                              min_fit_clients=config.server_config.num_clients,
                              fraction_fit=1.0)

    body_model = cnnTransformer(**config.model_config.body_config)

    tavit_server = TAViTServer(body=body_model,
                                 client_manager=client_manager, 
                                 strategy=strategy, 
                                 device=DEVICE,
                                 config=config.server_config,
                                 )

    app.start_tavit_server(server_address=args.server_address, 
                            server=tavit_server, 
                            config={"num_cycles":config.server_config.num_cycles})

if __name__=='__main__':
    main()
