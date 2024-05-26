# --- Imports --- #
import torch
import torch.nn as nn


class FreBlockftim(nn.Module):
    def __init__(self, nc):
        super(FreBlockftim, self).__init__()
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


class FreBlockfvim(nn.Module):
    def __init__(self, nc):
        super(FreBlockfvim, self).__init__()
        self.processreal = nn.Sequential(
            nn.Conv2d(nc, nc, kernel_size=1, padding=0, stride=1, groups=nc),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, kernel_size=(1,3), padding=(0,3//2), stride=1, groups=nc))
        self.processimag = nn.Sequential(
            nn.Conv2d(nc, nc, kernel_size=1, padding=0, stride=1, groups=nc),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, kernel_size=(1,3), padding=(0,3//2), stride=1, groups=nc))

    def forward(self, x):
        real = self.processreal(x.real)
        imag = self.processimag(x.imag)
        x_out = torch.complex(real, imag)

        return x_out


class FTIM(nn.Module):
    def __init__(self, in_nc):
        super(FTIM, self).__init__()

        self.frequency_process = FreBlockftim(in_nc)
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
        self.frequency_process = FreBlockfvim(in_nc)
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
        self.fvim_independence = '0'

    def forward(self, x):
        x = self.ftim(x)
        if self.fvim_independence == '1':
            x = self.cha(x)

        return x


from layers.RevIN import RevIN
class Model(nn.Module):
    def __init__(self,  config=None):
        super().__init__()

        self.e_layers = config.e_layers
        self.ProcessBlock = nn.ModuleList()
        for i in range(config.e_layers):
            self.ProcessBlock.append(ProcessBlock(config.d_model))

        self.conv = nn.Conv2d(1,config.d_model,kernel_size = (3, 1), padding = (3 // 2, 0))
        self.fc = nn.Sequential(
            nn.Linear(config.seq_len * config.d_model, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, config.pred_len)
        )
        self.revin_layer = RevIN(config.enc_in)

    def forward(self, x_enc):
        x_enc = self.revin_layer(x_enc, 'norm')

        B, T, N = x_enc.shape
        x_enc = torch.unsqueeze(x_enc,1)
        x_enc = self.conv(x_enc)
        for ii in range(self.e_layers):
            x_enc = self.ProcessBlock[ii](x_enc)

        x_enc = x_enc.permute(0,3,2,1)
        dec_out = self.fc(x_enc.reshape(B, N, -1)).permute(0, 2, 1)

        dec_out = self.revin_layer(dec_out, 'denorm')

        return dec_out
