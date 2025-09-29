import torch
import torch.nn as nn
import torch.nn.functional as F

from hm4.log_p_calc import log_normal_diag
from hm4.log_p_calc import log_standard_normal
from hm4.res_blocks import DecoderConvBlock
from hm4.res_blocks import DecoderUpscaleConvBlock
from hm4.res_blocks import EncoderConvBlock
from hm4.res_blocks import EncoderDownscaleConvBlock


INPUT_LEN = 248  # Only works with n_blocks=1


# Encoders
class EncoderX(nn.Module):
    def __init__(self, n_blocks=3, n_conv_layers=1, z_dim=16):
        """VAE Encoder for Noisy/Entangled/Remainder latent space.

        Parameters
        ----------
        n_blocks : int
            Number of downsmapling blocks in the encoder.
            Must be at least 1.
            Default is 3.
        n_conv_layers : int
            Number of convolutional blocks between each downsampling block.
            Default is 1.
        z_dim : int
            Dimension of the latent space.
            Default is 16.
        """
        super(EncoderX, self).__init__()
        self.n_blocks = n_blocks
        self.z_dim = z_dim
        # Calculate the shape of the flattened output

        # Number of channels is 2**(3+n_blocks) because we start with 1 channel
        # and double the number of channels with each block.
        # The length of the original input is INPUT_LEN,
        # and we halve it with each block.
        self.flatten_shape = 2 ** (3 + n_blocks) * (
            INPUT_LEN // (2**n_blocks) + min(1, n_blocks - 1)
        )

        self.conv_network = nn.ModuleList()
        self.conv_network.append(EncoderDownscaleConvBlock(1, 16, 5, 2))
        for i in range(n_conv_layers):
            self.conv_network.append(EncoderConvBlock(16, 16, 5, 2))

        for exp in range(4, self.n_blocks + 3):
            self.conv_network.append(
                EncoderDownscaleConvBlock(2**exp, 2 ** (exp + 1), 5, 2)
            )
            for i in range(n_conv_layers):
                self.conv_network.append(
                    EncoderConvBlock(2 ** (exp + 1), 2 ** (exp + 1), 5, 2)
                )

        self.mu = nn.Linear(self.flatten_shape, z_dim)
        self.log_var = nn.Linear(self.flatten_shape, z_dim)

        # Prevents crazy log_var values
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        "Encode input x into low dim latent space"
        output = x
        for conv_block in self.conv_network:
            output = conv_block(output)
        output = output.view(-1, self.flatten_shape)
        mu_e = self.mu(output)
        log_var_e = self.log_var(output)
        return mu_e, log_var_e


class EncoderY(nn.Module):
    def __init__(self, n_blocks=3, n_conv_layers=1, z_dim=16):
        """VAE Encoder for Clean/Disentangled latent space.

        Parameters
        ----------
        n_blocks : int
            Number of downsmapling blocks in the encoder.
            Must be at least 1.
            Default is 3.
        n_conv_layers : int
            Number of convolutional blocks between each downsampling block.
            Default is 1.
        z_dim : int
            Dimension of the latent space.
            Default is 16.
        """
        super(EncoderY, self).__init__()
        self.n_blocks = n_blocks
        self.z_dim = z_dim
        # Calculate the shape of the flattened output

        # Number of channels is 2**(3+n_blocks) because we start with 1 channel
        # and double the number of channels with each block.
        # The length of the original input is INPUT_LEN,
        # and we halve it with each block.
        self.flatten_shape = 2 ** (3 + n_blocks) * (
            INPUT_LEN // (2**n_blocks) + min(1, n_blocks - 1)
        )

        self.conv_network = nn.ModuleList()
        self.conv_network.append(EncoderDownscaleConvBlock(1, 16, 5, 2))
        for i in range(n_conv_layers):
            self.conv_network.append(EncoderConvBlock(16, 16, 5, 2))

        for exp in range(4, self.n_blocks + 3):
            self.conv_network.append(
                EncoderDownscaleConvBlock(2**exp, 2 ** (exp + 1), 5, 2)
            )
            for i in range(n_conv_layers):
                self.conv_network.append(
                    EncoderConvBlock(2 ** (exp + 1), 2 ** (exp + 1), 5, 2)
                )

        self.mu = nn.Linear(self.flatten_shape, z_dim)
        self.log_var = nn.Linear(self.flatten_shape, z_dim)

        # Prevents crazy log_var values
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        "Encode input x into low dim latent space"
        output = x
        for conv_block in self.conv_network:
            output = conv_block(output)
        output = output.view(-1, self.flatten_shape)
        mu_e = self.mu(output)
        log_var_e = self.log_var(output)
        return mu_e, log_var_e


# Decoders
class DecoderX(nn.Module):
    def __init__(self, n_blocks=3, n_conv_layers=1, z_dim=16):
        """Decoder for reconstructing input X.

        Parameters
        ----------
        n_blocks : int
            Number of downsmapling blocks in the encoder.
            Must be at least 1.
            Default is 3.
        n_conv_layers : int
            Number of convolutional blocks between each downsampling block.
            Default is 1.
        z_dim : int
            Dimension of the total latent spaces.
            Default is 32 (16 * 2 encoders).
        """
        super(DecoderX, self).__init__()
        self.n_blocks = n_blocks
        self.z_dim = z_dim
        # Calculate the shape of the flattened output
        self.flatten_shape = 2 ** (3 + n_blocks) * (
            INPUT_LEN // (2**n_blocks) + min(1, n_blocks - 1)
        )

        self.fc = nn.Linear(z_dim, self.flatten_shape)
        self.final_padding = 3 if n_blocks > 1 else 2

        self.trans_conv_network = nn.ModuleList()
        for exp in range(self.n_blocks + 3, 4, -1):
            self.trans_conv_network.append(
                DecoderUpscaleConvBlock(2**exp, 2 ** (exp - 1), 5, 2, 1)
            )
            for i in range(n_conv_layers):
                self.trans_conv_network.append(
                    DecoderConvBlock(2 ** (exp - 1), 2 ** (exp - 1), 5, 2)
                )

        self.trans_conv_network.append(
            DecoderUpscaleConvBlock(16, 1, 5, self.final_padding, 1)
        )
        for i in range(n_conv_layers):
            self.trans_conv_network.append(DecoderConvBlock(1, 1, 5, 2))
        self.final_layer = nn.Conv1d(1, 1, 1, 1, 0)

        # Prevents crazy log_var values
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        self.final_layer.weight.data.fill_(0)
        self.final_layer.bias.data.fill_(0)

    def forward(self, zx, zy):
        # Concatenate the latent spaces together
        zx_zy = torch.cat((zx, zy), dim=1)
        output = self.fc(zx_zy)
        # Reshape output to correct features and channels
        # for the number of blocks
        output = output.view(
            -1,
            2 ** (3 + self.n_blocks),
            (INPUT_LEN // (2 ** (self.n_blocks)) + min(1, self.n_blocks - 1)),
        )
        for trans_conv_block in self.trans_conv_network:
            output = trans_conv_block(output)
        output = self.final_layer(output)
        return output


class Classifier(nn.Module):
    def __init__(self, n_classes=38, z_dim=16):
        """Classifier for predicting the mineral class of the input spectra.

        Parameters
        ----------
        n_classes : int
            Number of classes to predict.
            Default is 37.
        z_dim : int
            Dimension of the disentangled latent space.
            Default is 16.
        """
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(z_dim, 64)
        self.fc2 = nn.Linear(64, n_classes)
        self.relu = nn.LeakyReLU()

    def forward(self, zy):
        output = self.fc1(zy)
        output = self.relu(output)
        output = self.fc2(output)
        return output


class Prior(nn.Module):
    def __init__(self, latent_dim):
        """Prior distribution for the latent space.
        Currently set to a standard normal distribution.
        """
        super(Prior, self).__init__()
        self.latent_dim = latent_dim

    def sample(self, batch_size):
        return torch.randn((batch_size, self.latent_dim))

    def log_prob(self, z):
        return log_standard_normal(z)


class VAEClassifier(nn.Module):
    def __init__(
        self, n_blocks=3, n_conv_layers=1, zx_dim=16, zy_dim=16, n_classes=38
    ):
        """Hybrid VAE and Classifier network.

        Parameters
        ----------
        n_blocks : int
            Number of convolutional up and downsampling blocks
            in the encoder and decoder.
            Must be at least 1, and less than 6.
            Default = 3.
        n_conv_layers : int
            Number of convolutional blocks between each downsampling block.
            Default is 1.
        zx_dim : int
            Dimension of the noisy/entangled latent space.
            Default is 16.
        zy_dim : int
            Dimension of the clean/disentangled latent space.
            Default is 16.
        n_classes : int
            Number of classes to predict.
            Default is 37.
        """
        super(VAEClassifier, self).__init__()
        self.n_blocks = n_blocks
        self.n_conv_layers = n_conv_layers
        self.zx_dim = zx_dim
        self.zy_dim = zy_dim
        self.n_classes = n_classes

        self.encoder_x = EncoderX(
            self.n_blocks, self.n_conv_layers, self.zx_dim
        )
        self.encoder_y = EncoderY(
            self.n_blocks, self.n_conv_layers, self.zy_dim
        )
        self.decoder = DecoderX(
            self.n_blocks, self.n_conv_layers, self.zx_dim + self.zy_dim
        )
        self.classifier = Classifier(self.n_classes, self.zy_dim)
        self.prior_x = Prior(self.zx_dim)
        self.prior_y = Prior(self.zy_dim)

    def forward(self, x, y):
        # Encode
        mu_e_x, log_var_e_x = self.encoder_x(x)
        mu_e_y, log_var_e_y = self.encoder_y(x)

        # Reparameterization Trick
        zx = mu_e_x + torch.exp(0.5 * log_var_e_x) * torch.randn_like(mu_e_x)
        zy = mu_e_y + torch.exp(0.5 * log_var_e_y) * torch.randn_like(mu_e_y)

        # Decode
        x_recon = self.decoder(zx, zy)
        y_recon = self.classifier(zy)
        return (
            x_recon,
            y_recon,
            mu_e_x,
            log_var_e_x,
            mu_e_y,
            log_var_e_y,
            zx,
            zy,
        )

    def loss_function(self, x, y):
        """
        Calculate loss terms.
        Also return y_recon, for calculating classification metrics
        """
        x_recon, y_recon, mu_e_x, log_var_e_x, mu_e_y, log_var_e_y, zx, zy = (
            self.forward(x, y)
        )

        # Losses
        # Reconstruction losses
        RE_x = F.mse_loss(x_recon, x)
        RE_y = F.cross_entropy(y_recon, y.long().squeeze())

        # KL Divergence
        KL_x = (
            self.prior_x.log_prob(zx)
            - log_normal_diag(zx, mu_e_x, log_var_e_x)
        ).sum(-1)
        KL_y = (
            self.prior_y.log_prob(zy)
            - log_normal_diag(zy, mu_e_y, log_var_e_y)
        ).sum(-1)

        return RE_x, RE_y, KL_x, KL_y, y_recon

    def classify(self, x):
        """Classify input x into a mineral class.

        Parameters
        ----------
        x : torch.Tensor
            Input spectra to classify.

        Returns
        -------
        y_pred : torch.Tensor
            Predicted mineral class probabilities.
        """
        mu_e_y, log_var_e_y = self.encoder_y(x)
        # Use the mean rather than sampling
        y_pred = self.classifier(mu_e_y)
        y_pred = F.softmax(y_pred, dim=1)
        return y_pred

    def reconstruct(self, x):
        """Reconstruct input x.

        Parameters
        ----------
        x : torch.Tensor
            Input spectra to reconstruct.

        Returns
        -------
        x_recon : torch.Tensor
            Reconstructed input spectra.
        """
        mu_e_x, log_var_e_x = self.encoder_x(x)
        mu_e_y, log_var_e_y = self.encoder_y(x)
        zx = mu_e_x + torch.exp(0.5 * log_var_e_x) * torch.randn_like(mu_e_x)
        zy = mu_e_y + torch.exp(0.5 * log_var_e_y) * torch.randn_like(mu_e_y)
        x_recon = self.decoder(zx, zy)
        return x_recon

    def reconstruct_classify(self, x):
        """Run a full forward pass on input x.

        Parameters
        ----------
        x : torch.Tensor
            Input spectra to run through the network.

        Returns
        -------
        x_recon : torch.Tensor
            Reconstructed input spectra.
        y_pred : torch.Tensor
            Predicted mineral class probabilities.
        """
        mu_e_x, log_var_e_x = self.encoder_x(x)
        mu_e_y, log_var_e_y = self.encoder_y(x)
        zx = mu_e_x + torch.exp(0.5 * log_var_e_x) * torch.randn_like(mu_e_x)
        zy = mu_e_y + torch.exp(0.5 * log_var_e_y) * torch.randn_like(mu_e_y)
        x_recon = self.decoder(zx, zy)
        y_pred = self.classifier(mu_e_y)
        y_pred = F.softmax(y_pred, dim=1)
        return x_recon, y_pred
