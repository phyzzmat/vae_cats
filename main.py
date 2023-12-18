from pathlib import Path
import wandb
from train import train_model
from model import *
import argparse
import json
from torchvision.io import read_image
from torch.utils.data import TensorDataset, DataLoader
import os
from piq import FID


def main(args):
    config = json.loads(open(args.config).read())
    wandb.init(
        project="Cats",
        config=config
    )
    data = [
        read_image('dataset/cats/' + str(file)).tolist() for file in sorted(os.listdir('dataset/cats'))
    ]
    TRAIN_SIZE = .8
    train_data_ = data[:int(TRAIN_SIZE * len(data))]
    test_data_ = data[int(TRAIN_SIZE * len(data)):]
    td = torch.tensor(train_data_).to('cuda').float().reshape(-1, 3 * 64 * 64) / 255.
    td = torch.cat([td, torch.flip(td.reshape(-1, 3, 64, 64), dims=(-1,)).reshape(-1, 3 * 64 * 64)])
    train_data = TensorDataset(td)
    test_data = TensorDataset(torch.tensor(test_data_).to('cuda').reshape(-1, 3, 64, 64).float().reshape(-1, 3 * 64 * 64) / 255.)
    fid_metric = FID()
    test_fid_dl = DataLoader(TensorDataset(torch.tensor(test_data_).to('cuda').reshape(-1, 3, 64, 64).float() / 255.), collate_fn=lambda x: {'images': torch.stack(x[0])})
    test_feats = fid_metric.compute_feats(test_fid_dl)
    vae = VAE(3, config["latent_dim"], 64 * 64, config["beta"])
    train_model(vae.parameters(), test_feats, fid_metric, vae, vae.batch_vlb, train_data,
                               num_epochs=1000, learning_rate=config["lr"], batch_size=config["batch_size"])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    main(args)