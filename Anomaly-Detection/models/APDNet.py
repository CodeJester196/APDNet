# --- Imports --- #
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FreBlockSpa(nn.Module):
    def __init__(self, nc):
        super(FreBlockSpa, self).__init__()
        self.processreal = nn.Sequential(
            nn.Conv2d(nc, nc, kernel_size=(3,1), padding=(3//2,0), stride=1, groups=nc),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, kernel_size=(7,1), padding=(7//2,0), stride=1, groups=nc))
        self.processimag = nn.Sequential(
            nn.Conv2d(nc, nc, kernel_size=(3,1), padding=(3//2,0), stride=1, groups=nc),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, kernel_size=(7,1), padding=(7//2,0), stride=1, groups=nc))

    def forward(self, x):
        real = self.processreal(x.real)
        imag = self.processimag(x.imag)
        x_out = torch.complex(real, imag)
        return x_out

class FreBlockCha(nn.Module):
    def __init__(self, nc):
        super(FreBlockCha, self).__init__()
        self.processreal = nn.Sequential(
            nn.Conv2d(nc, nc, kernel_size=1, padding=0, stride=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, kernel_size=(1,3), padding=(0,3//2), stride=1))
        self.processimag = nn.Sequential(
            nn.Conv2d(nc, nc, kernel_size=1, padding=0, stride=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, kernel_size=(1,3), padding=(0,3//2), stride=1))

    def forward(self, x):
        real = self.processreal(x.real)
        imag = self.processimag(x.imag)
        x_out = torch.complex(real, imag)

        return x_out


class FTIM(nn.Module):
    def __init__(self, in_nc):
        super(FTIM, self).__init__()
        self.frequency_process = FreBlockSpa(in_nc)
        self.frequency_spatial = nn.Conv2d(in_nc, in_nc, (3,1), 1, (3//2,0))
        self.cat = nn.Conv2d(2 * in_nc, in_nc, 3, 1, 1)

    def forward(self, x):
        xori = x
        B , C , L , N  = x.shape
        x_freq = torch.fft.rfft(x,dim=2, norm='ortho')
        x_freq = self.frequency_process(x_freq)
        x_freq_spatial = torch.fft.irfft(x_freq, n=L,dim=2, norm='ortho')
        x_freq_spatial = self.frequency_spatial(x_freq_spatial)
        return x_freq_spatial+xori


class FVIM(nn.Module):
    def __init__(self, in_nc):
        super(FVIM, self).__init__()
        self.frequency_process = FreBlockCha(in_nc)
        self.frequency_spatial = nn.Conv2d(in_nc, in_nc, 1, 1, 0)
        self.cat = nn.Conv2d(2 * in_nc, in_nc, 1, 1, 0)

    def forward(self, x):
        xori = x
        B, C, L, N = x.shape
        x_freq = torch.fft.rfft(x,  dim=-1, norm='ortho')
        x_freq = self.frequency_process(x_freq)
        x_freq_spatial = torch.fft.irfft(x_freq, n=N,  dim=-1, norm='ortho')
        x_freq_spatial = self.frequency_spatial(x_freq_spatial)
        return x_freq_spatial + xori


class ProcessBlock(nn.Module):
    def __init__(self, nc):
        super(ProcessBlock, self).__init__()
        self.ftim = FTIM(nc)
        self.fvim = FVIM(nc)
        self.fvim_independence = '1'

    def forward(self, x):
        x = self.ftim(x)
        if self.fvim_independence == '1':
            x = self.fvim(x)

        return x



# from layers.RevIN import RevIN
class Model(nn.Module):
    def __init__(self,  config=None,nc = 32):
        super().__init__()
        self.task_name = config.task_name
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.e_layers = config.e_layers
        self.ProcessBlock = nn.ModuleList()
        for i in range(config.e_layers):
            self.ProcessBlock.append(ProcessBlock(config.d_model))
        self.linear = nn.Linear(config.d_model,1)
        self.conv1 = nn.Conv2d(1,config.d_model,(7,1),1,padding=(7//2,0))
        self.conv2 = nn.Conv2d(1, config.d_model, (1,3), 1,padding=(0,3//2))
        self.conv = nn.Conv2d(2*config.d_model,config.d_model,1,1)
        self.fc = nn.Sequential(
            nn.Linear(config.enc_in * config.d_model, 1024),
            nn.ReLU(),
            nn.Linear(1024, config.enc_in)
        )

    def forward(self, x_enc):
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
            B, T, N = x_enc.shape
            x_enc = torch.unsqueeze(x_enc, 1)
            x_enc1 = self.conv1(x_enc)
            x_enc2 = self.conv2(x_enc)
            x_enc = self.conv(torch.cat((x_enc1, x_enc2), dim=1))
            for ii in range(self.e_layers):
                x_enc = self.ProcessBlock[ii](x_enc)
            x_enc = x_enc.permute(0, 2, 3, 1)
            dec_out = self.fc(x_enc.reshape(B, T, -1))
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * \
                      (stdev[:, 0, :].unsqueeze(1).repeat(
                          1, self.pred_len + self.seq_len, 1))
            dec_out = dec_out + \
                      (means[:, 0, :].unsqueeze(1).repeat(
                          1, self.pred_len + self.seq_len, 1))

            return dec_out  # [B, L, D]




