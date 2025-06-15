import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(
        self,
        backbone,
        encoder_dim=512,
        decoder_channels=512,
        decoder_h=8,
        decoder_w=8,
    ):
        """Autoencoder with a backbone and a transposed convolution decoder.
        Args:
            backbone (nn.Module): Pretrained  backbone.
            encoder_dim (int): Dimension of the encoder output.
            decoder_channels (int): Number of channels in the decoder output.
            decoder_h (int): Height of the decoder output feature map.
            decoder_w (int): Width of the decoder output feature map.

        'encoder_dim' and 'decoder_channels' can vary based on the backbone used
        and the compression ratio desired.
        """
        super().__init__()
        self.encoder = backbone
        self.decoder_channels = decoder_channels
        self.decoder_h = decoder_h
        self.decoder_w = decoder_w

        # Project from encoder_dim to decoder_channels * decoder_h * decoder_w
        # ReLU + BatchNorm1d maintains stable activation and improves training.
        # Dropout helps avoid overfitting of the head, which is important since the backbone is frozen.
        self.bottleneck_layer = nn.Sequential(
            nn.Linear(encoder_dim, decoder_channels),
            nn.ReLU(),
            nn.BatchNorm1d(decoder_channels),
            nn.Dropout(0.5),
        )

        self.project = nn.Linear(
            decoder_channels, decoder_channels * decoder_h * decoder_w
        )

        # Decoder: input channels = decoder_channels
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                decoder_channels, 256, 3, stride=2, padding=1, output_padding=1
            ),  # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(
                256, 128, 3, stride=2, padding=1, output_padding=1
            ),  # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(
                128, 64, 3, stride=2, padding=1, output_padding=1
            ),  # 32x32 -> 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 32, 3, stride=2, padding=1, output_padding=1
            ),  # 64x64 -> 128x128
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 3, 4, stride=2, padding=1, output_padding=0
            ),  # 128x128 -> 256x256
            nn.Tanh(),
        )

    def encode(self, x):
        x = self.encoder(x)  # [batch, encoder_dim]
        x = self.bottleneck_layer(x)
        x = self.project(
            x
        )  # [batch, decoder_channels * decoder_h * decoder_w]
        x = x.view(
            x.size(0), self.decoder_channels, self.decoder_h, self.decoder_w
        )
        return x

    def decode(self, x):
        x = self.decoder(x)
        x = nn.functional.interpolate(
            x, size=(224, 224), mode='bilinear', align_corners=False
        )
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck_layer(x)
        return x
