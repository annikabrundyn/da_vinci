from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F

from data.right_data import RightDaVinciDataModule
from models.right_view.vae.vae_components import Encoder, Decoder


class VAE(pl.LightningModule):

    def __init__(
        self,
        frames_per_sample: int,
        frames_to_drop: int,
        input_height: int = 192,
        input_width: int = 384,
        output_height: int = 192,
        output_width: int = 384,
        enc_out_dim: int = 512,
        kl_coeff: float = 0.1,
        latent_dim: int = 256,
        in_channels: int = 3,
        lr: float = 1e-4,
        **kwargs
    ):
        """
        Args:
            input_height: height of the images
            enc_out_dim: set according to the out_channel count of
                encoder used (512 for resnet18, 2048 for resnet50)
            kl_coeff: coefficient for kl term of the loss
            latent_dim: dim of latent space
            lr: learning rate for Adam
        """

        super(VAE, self).__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.kl_coeff = kl_coeff
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.in_channels = 3

        self.input_height = input_height
        self.input_width = input_width

        self.output_height = output_height
        self.output_width = output_width

        self.out_channels = 3


        self.encoder = self.init_encoder(self.enc_out_dim, self.latent_dim,
                                         self.in_channels, self.input_height, self.input_width)
        self.decoder = self.init_decoder(self.enc_out_dim, self.latent_dim,
                                         self.out_channels, self.output_height, self.output_width)

        self.fc_mu = nn.Linear(self.enc_out_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.enc_out_dim, self.latent_dim)

    def init_encoder(self, hidden_dim, latent_dim, in_channels, input_height, input_width):
        encoder = Encoder(hidden_dim, latent_dim, in_channels, input_height, input_width)
        return encoder

    def init_decoder(self, hidden_dim, latent_dim, in_channels, output_height, output_width):
        decoder = Decoder(hidden_dim, latent_dim, in_channels, output_height, output_width)
        return decoder

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return self.decoder(z)

    def _run_step(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return z, self.decoder(z), p, q

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def step(self, batch, batch_idx):
        x, y = batch
        z, x_hat, p, q = self._run_step(x)

        recon_loss = F.mse_loss(x_hat, x, reduction='mean')

        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = log_qz - log_pz
        kl = kl.mean()
        kl *= self.kl_coeff

        loss = kl + recon_loss

        logs = {
            "recon_loss": recon_loss,
            "kl": kl,
            "loss": loss,
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict(
            {f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--lr", type=float, default=1e-4)

        parser.add_argument("--enc_out_dim", type=int, default=512)
        parser.add_argument("--kl_coeff", type=float, default=0.1)
        parser.add_argument("--latent_dim", type=int, default=256)

        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--data_dir", type=str, default="/Users/annikabrundyn/Developer/da_vinci/daVinci_data")
        parser.add_argument('--input_height', type=int, default=192)
        parser.add_argument('--input_width', type=int, default=384, help='input image width')
        parser.add_argument('--output_height', type=int, default=192)
        parser.add_argument('--output_width', type=int, default=384)
        parser.add_argument('--in_channels', type=int, default=3)

        return parser



if __name__ == "__main__":
    # sets seed for numpy, torch, python.random and PYTHONHASHSEED
    print("start right direct model")
    pl.seed_everything(42)

    parser = ArgumentParser()

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # model args
    parser = VAE.add_model_specific_args(parser)
    args = parser.parse_args()

    # data
    dm = RightDaVinciDataModule(
        args.data_dir,
        frames_per_sample=1,
        frames_to_drop=0,
        is_color_input=True,
        is_color_output=True,
        extra_info=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dm.setup()
    print("dm setup")

    # sanity check
    print("size of trainset:", len(dm.train_samples))
    print("size of validset:", len(dm.val_samples))
    print("size of testset:", len(dm.test_samples))

    img, target = next(iter(dm.train_dataloader()))
    print(img.shape)
    print(target.shape)

    # model
    model = VAE(**args.__dict__)
    print("model instance created")
    print("lightning version", pl.__version__)

    # train
    trainer = pl.Trainer.from_argparse_args(args)
    print("trainer created")
    trainer.fit(model, dm)