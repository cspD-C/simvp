import torch
from torch import nn

from .blocks import ConvSC, Inception


def stride_generator(depth, reverse=False):
    strides = [1, 2] * 10
    if reverse:
        return list(reversed(strides[:depth]))
    return strides[:depth]


class Encoder(nn.Module):
    def __init__(self, c_in, c_hid, n_s):
        super().__init__()
        strides = stride_generator(n_s)
        self.enc = nn.Sequential(
            ConvSC(c_in, c_hid, stride=strides[0]),
            *[ConvSC(c_hid, c_hid, stride=s) for s in strides[1:]],
        )

    def forward(self, x):
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1


class Decoder(nn.Module):
    def __init__(self, c_hid, c_out, n_s):
        super().__init__()
        strides = stride_generator(n_s, reverse=True)
        self.dec = nn.Sequential(
            *[
                ConvSC(c_hid, c_hid, stride=s, transpose=True)
                for s in strides[:-1]
            ],
            ConvSC(2 * c_hid, c_hid, stride=strides[-1], transpose=True),
        )
        self.readout = nn.Conv2d(c_hid, c_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec) - 1):
            hid = self.dec[i](hid)
        y = self.dec[-1](torch.cat([hid, enc1], dim=1))
        y = self.readout(y)
        return y


class MidXNet(nn.Module):
    def __init__(self, channel_in, channel_hid, n_t, incep_ker=(3, 5, 7, 11), groups=8):
        super().__init__()
        self.n_t = n_t
        enc_layers = [
            Inception(
                channel_in,
                channel_hid // 2,
                channel_hid,
                incep_ker=incep_ker,
                groups=groups,
            )
        ]
        for _ in range(1, n_t - 1):
            enc_layers.append(
                Inception(
                    channel_hid,
                    channel_hid // 2,
                    channel_hid,
                    incep_ker=incep_ker,
                    groups=groups,
                )
            )
        enc_layers.append(
            Inception(
                channel_hid,
                channel_hid // 2,
                channel_hid,
                incep_ker=incep_ker,
                groups=groups,
            )
        )

        dec_layers = [
            Inception(
                channel_hid,
                channel_hid // 2,
                channel_hid,
                incep_ker=incep_ker,
                groups=groups,
            )
        ]
        for _ in range(1, n_t - 1):
            dec_layers.append(
                Inception(
                    2 * channel_hid,
                    channel_hid // 2,
                    channel_hid,
                    incep_ker=incep_ker,
                    groups=groups,
                )
            )
        dec_layers.append(
            Inception(
                2 * channel_hid,
                channel_hid // 2,
                channel_in,
                incep_ker=incep_ker,
                groups=groups,
            )
        )

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        bsz, steps, chans, height, width = x.shape
        x = x.reshape(bsz, steps * chans, height, width)

        skips = []
        z = x
        for i in range(self.n_t):
            z = self.enc[i](z)
            if i < self.n_t - 1:
                skips.append(z)

        z = self.dec[0](z)
        for i in range(1, self.n_t):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1))

        return z.reshape(bsz, steps, chans, height, width)


class SimVP(nn.Module):
    def __init__(
        self,
        shape_in,
        hid_s=16,
        hid_t=256,
        n_s=4,
        n_t=8,
        incep_ker=(3, 5, 7, 11),
        groups=8,
    ):
        super().__init__()
        t, c, _, _ = shape_in
        self.enc = Encoder(c, hid_s, n_s)
        self.hid = MidXNet(t * hid_s, hid_t, n_t, incep_ker, groups)
        self.dec = Decoder(hid_s, c, n_s)

    def forward(self, x_raw):
        bsz, steps, channels, height, width = x_raw.shape
        x = x_raw.view(bsz * steps, channels, height, width)

        embed, skip = self.enc(x)
        _, c_hid, h_hid, w_hid = embed.shape

        z = embed.view(bsz, steps, c_hid, h_hid, w_hid)
        hid = self.hid(z)
        hid = hid.reshape(bsz * steps, c_hid, h_hid, w_hid)

        y = self.dec(hid, skip)
        return y.reshape(bsz, steps, channels, height, width)
