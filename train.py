from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import optim
import wandb
from model import show_images
import torch
from torch.utils.data import TensorDataset
from piq import FID, SSIMLoss

def train_model(parameters, fid_test_feats, fid, model, batch_loss, data,
                batch_size=64, num_epochs=5,
                learning_rate=1e-3):
    gd = optim.Adam(parameters, lr=learning_rate, weight_decay=0.01)
    train_losses = []
    step = 0
    fid = FID()
    for epoch in range(num_epochs):
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
        with tqdm(enumerate(dataloader), total=len(dataloader)) as iterator:
            for _, batch in iterator:
                step += 1
                if len(batch) == 1:
                    batch = batch[0]
                gd.zero_grad()
                loss, rec, kl = batch_loss(batch)
                (-loss).backward()
                wandb.log(
                    {
                        "training loss": (-loss).item(),
                        "reconstruction loss": (-rec).item(),
                        "KL-divergence": (-kl).item()
                    }, step=step
                )
                gd.step()
                train_losses.append(float(loss))            
                iterator.set_description('Train loss: %.3f, rec loss: %.3f, kl loss: %3f' % (loss.item(), rec.item(), kl.item()))
            if epoch % 10 == 0:
                generated = model.generate_samples(100)
                imgs = show_images(generated.detach().cpu())
                our_imgs_dl = DataLoader(TensorDataset(generated.reshape(100, 3, 64, 64)), collate_fn=lambda x: {'images': torch.stack(x[0])})
                our_imgs_feats = fid.compute_feats(our_imgs_dl)
                ssim = SSIMLoss()
                wandb.log({
                    "images": wandb.Image(imgs),
                    "FID": fid(our_imgs_feats, fid_test_feats),
                    "SSIM": ssim(generated.reshape(100, 3, 64, 64), data.tensors[0][:100].reshape(100, 3, 64, 64))
                }, step=step)
                torch.save(model.state_dict(), f'model_{epoch}')