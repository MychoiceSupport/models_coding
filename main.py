import argparse
import torch
import datetime
import json
import yaml
import os
import logging
import numpy as np

from traj_dataset import get_the_dataloader
from utils import *
from models import *

def main(args):
    SEED = args.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    path = 'config/' + args.config
    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    config['seed'] = SEED

    print(json.dumps(config, indent=4))
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    foldername = (
        "./save/gps_" + current_time + "/"
    )

    print('model folder:', foldername)
    os.makedirs(foldername, exist_ok=True)
    with open(foldername + "config.json", "w") as f:
        json.dump(config, f, indent=4)

    train_loader, val_loader, test_loader, scaler = get_the_dataloader(
        args.batch_size, target_dim=config['diffusion']['input_dim'],device=args.device,
        num_workers=args.num_workers
    )
    if args.model_type == 'TCN':
        model = TCN(config['diffusion'], args, inputdim=2, seq_len=11, device=args.device).to(args.device)
    elif args.model_type == 'BiLSTM':
        model = BiLSTM(config['diffusion'], args, inputdim=2, seq_len=11, device=args.device).to(args.device)
    elif args.model_type == 'GRU':
        model = GRU(config['diffusion'], args, inputdim=2, seq_len=11, device=args.device).to(args.device)
    elif args.model_type == 'fc':
        model = FC(config['diffusion'], args, inputdim=2, seq_len=11, device=args.device).to(args.device)
    elif args.model_type == 'Att':
        model = Att(config['diffusion'], args, inputdim=2, seq_len=11, device=args.device).to(args.device)
    else:
        raise Exception('You should input the right model!')
    if args.modelfolder == '':
        train(
            model,args,
            config['train'],
            train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            scaler=scaler,
            foldername = foldername
        )
    else:
        model.load_state_dict(torch.load('./save/{}'.format(args.foldername)+ "/model.pth", map_location=args.device))

    logging.basicConfig(filename=foldername + '/test_model.log', level=logging.DEBUG)
    logging.info('model_name = {}'.format(args.modelfolder))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model_Selection')
    parser.add_argument('--config', type=str, default='base.yaml')
    parser.add_argument('--device', default='cpu' if torch.cuda.is_available() else 'cpu', help='Device for Attack')
    parser.add_argument('--num_workers', type=int, default=0, help='Device for Attack')
    parser.add_argument('--modelfolder', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_type', type=str, default='Att')
    parser.add_argument('--channels', type=int, default=128)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--Bi', type=bool, default=False)
    parser.add_argument('--learning_rate', type=float, default=False)
    parser.add_argument('--nheads', type=int, default=8)
    parser.add_argument('--timeemb', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--side_channel', type=int, default=64)

    args = parser.parse_args()
    print(args)
    main(args)