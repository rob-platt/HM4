import torch.nn.functional as F
from torch import nn


class EncoderDownscaleConvBlock(nn.Module):
    """Encoder Convolutional Block, with 2 convolutional layers and batch
    normalization. Uses a resiudal connection.
    Downsamples input by a factor of 2.
    Uses LeakyReLU activation function.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(EncoderDownscaleConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        self.conv_padding = self.kernel_size // 2
        # resize convolution
        self.conv1 = nn.Conv1d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            stride=2,
            padding=self.padding,
        )
        self.relu = nn.LeakyReLU()
        self.batch_norm1 = nn.BatchNorm1d(self.out_channels)
        # normal conv layer
        self.conv2 = nn.Conv1d(
            self.out_channels,
            self.out_channels,
            self.kernel_size,
            stride=1,
            padding=self.conv_padding,
        )
        self.batch_norm2 = nn.BatchNorm1d(self.out_channels)

        self.residual_conv = nn.Conv1d(
            self.in_channels,
            self.out_channels,
            kernel_size=1,
            stride=1,
        )

        # Prevents crazy log_var values
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        h = self.conv1(x)
        h = self.batch_norm1(h)
        h = self.relu(h)

        h = self.conv2(h)
        h = self.batch_norm2(h)
        h = self.relu(h)

        residual = self.residual_conv(x)
        residual = F.interpolate(residual, size=h.size(-1), mode="linear")
        return h + residual


class EncoderConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(EncoderConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        self.conv1 = nn.Conv1d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            stride=1,
            padding=self.padding,
        )
        self.relu = nn.LeakyReLU()
        self.batch_norm1 = nn.BatchNorm1d(self.out_channels)
        self.conv2 = nn.Conv1d(
            self.out_channels,
            self.out_channels,
            self.kernel_size,
            stride=1,
            padding=self.padding,
        )
        self.batch_norm2 = nn.BatchNorm1d(self.out_channels)

        self.residual_conv = nn.Conv1d(
            self.in_channels,
            self.out_channels,
            kernel_size=1,
            stride=1,
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        h = self.conv1(x)
        h = self.batch_norm1(h)
        h = self.relu(h)

        h = self.conv2(h)
        h = self.batch_norm2(h)
        h = self.relu(h)

        residual = self.residual_conv(x)
        return h + residual


class DecoderUpscaleConvBlock(nn.Module):
    """Decoder Convolutional Block, with 2 convolutional layers and batch
    normalization. Uses a resiudal connection.
    Upsamples input by a factor of 2 via transposed convolution.
    Uses LeakyReLU activation function.
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, padding, output_padding=0
    ):
        super(DecoderUpscaleConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.output_padding = output_padding

        self.conv_padding = self.kernel_size // 2

        # resize convolution
        self.ConvTranspose1 = nn.ConvTranspose1d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            stride=2,
            padding=self.padding,
            output_padding=self.output_padding,
        )
        self.relu = nn.LeakyReLU()
        self.batch_norm1 = nn.BatchNorm1d(self.out_channels)
        # normal conv layer
        self.conv2 = nn.Conv1d(
            self.out_channels,
            self.out_channels,
            self.kernel_size,
            stride=1,
            padding=self.conv_padding,
        )
        self.batch_norm2 = nn.BatchNorm1d(out_channels)

        self.residual_conv = nn.Conv1d(
            self.in_channels,
            self.out_channels,
            kernel_size=1,
            stride=1,
        )

        # Prevents crazy log_var values
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        h = self.ConvTranspose1(x)
        h = self.batch_norm1(h)
        h = self.relu(h)

        h = self.conv2(h)
        h = self.batch_norm2(h)
        h = self.relu(h)

        residual = self.residual_conv(x)
        residual = F.interpolate(residual, size=h.size(-1), mode="linear")
        return h + residual


class DecoderConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(DecoderConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        self.conv1 = nn.Conv1d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            stride=1,
            padding=self.padding,
        )
        self.relu = nn.LeakyReLU()
        self.batch_norm1 = nn.BatchNorm1d(self.out_channels)
        self.conv2 = nn.Conv1d(
            self.out_channels,
            self.out_channels,
            self.kernel_size,
            stride=1,
            padding=self.padding,
        )
        self.batch_norm2 = nn.BatchNorm1d(self.out_channels)

        self.residual_conv = nn.Conv1d(
            self.in_channels,
            self.out_channels,
            kernel_size=1,
            stride=1,
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        h = self.conv1(x)
        h = self.batch_norm1(h)
        h = self.relu(h)

        h = self.conv2(h)
        h = self.batch_norm2(h)
        h = self.relu(h)

        residual = self.residual_conv(x)
        return h + residual
