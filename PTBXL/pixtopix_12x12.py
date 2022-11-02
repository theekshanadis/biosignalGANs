from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
from sklearn.model_selection import KFold
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchaudio
from torchvision import transforms
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib
from torch import nn
from torch.autograd import Variable
import torchvision

plt.style.use('ggplot')
np.seterr(divide='ignore', invalid='ignore')


class Dataset1x2_Rec(torch.utils.data.Dataset):
    def __init__(self, indexes, test=False):
        self.indexes = indexes
        self.stft_f = torchaudio.transforms.Spectrogram(win_length=6, n_fft=255, power=None, return_complex=True)

    def __getitem__(self, index):
        x = np.load(self.indexes[index]).T

        x_spec = self.stft_f(torch.from_numpy(x).float())
        spec_mag, spec_ang = x_spec.abs(), x_spec.angle()

        spec_mag = F.pad(spec_mag, pad=(14, 14), mode='constant', value=0)
        spec_ang = F.pad(spec_ang, pad=(14, 14), mode='constant', value=0)

        ecg = torch.cat([spec_mag, spec_ang], dim=0)
        return ecg, ecg

    def __len__(self):
        return self.indexes.shape[0]

class Dataset1x2_Rec_(torch.utils.data.Dataset):
    def __init__(self, indexes, test=False):
        self.indexes = indexes
        self.stft_f = torchaudio.transforms.Spectrogram(win_length=6, n_fft=255, power=None, return_complex=True)

    def __getitem__(self, index):
        x = np.load(self.indexes[index]).T

        x_spec = self.stft_f(torch.from_numpy(x).float())
        spec_mag, spec_ang = x_spec.abs(), x_spec.angle()

        spec_mag = F.pad(spec_mag, pad=(14, 14), mode='constant', value=0)
        spec_ang = F.pad(spec_ang, pad=(14, 14), mode='constant', value=0)

        ecg = torch.cat([spec_mag, spec_ang], dim=0)

        return ecg, ecg, int(self.indexes[index].split('_')[1])

    def __len__(self):
        return self.indexes.shape[0]

class Conv2dEncoder_v1(torch.nn.Module):
    def __init__(self):
        super(Conv2dEncoder_v1, self).__init__()

        self.model = nn.Sequential(
            self.conv_layer(12 * 2, 24, stride=1, normalize=False),
            self.conv_layer(24, 24, down=False),
            self.conv_layer(24, 32),
            self.conv_layer(32, 32),
            self.conv_layer(32, 36),
            self.conv_layer(36, 48),
            self.conv_layer(48, 64, down=False),
            self.conv_layer(64, 72, dropout=0.5),
            self.conv_layer(72, 84, dropout=0.5, down=False),
            self.conv_layer(84, 96, dropout=0.5, down=False),
            self.conv_layer(96, 128, dropout=0.5, normalize=False),
        )

    def conv_layer(self, in_channel, out_channel, stride=2, normalize=True, dropout=0.0, down=True):
        if down:
            layers = [nn.Conv2d(in_channel, out_channel, (3 if stride == 1 else 4, 4), (stride, 2), (1, 1), bias=False)]
        else:
            layers = [nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channel))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x.flatten(start_dim=1)

class Conv2dDecoder_v1(torch.nn.Module):
    def __init__(self):
        super(Conv2dDecoder_v1, self).__init__()
        self.model = nn.Sequential(
            self.conv_layer(128, 96, dropout=0.5, normalize=False),
            self.conv_layer(96, 84, dropout=0.5, up=False),
            self.conv_layer(84, 72, dropout=0.5),
            self.conv_layer(72, 64, dropout=0.5),
            self.conv_layer(64, 48, up=False),
            self.conv_layer(48, 36, dropout=0.5),
            self.conv_layer(36, 32, dropout=0.5),
            self.conv_layer(32, 32, up=False),
            self.conv_layer(32, 24),
            self.conv_layer(24, 24, up=False),
            nn.ConvTranspose2d(24, 24, (3, 4), (1, 2), 1)
        )

    def conv_layer(self, in_channels, out_channels, stride=2, dropout=0.0, normalize=True, up=True):
        if up:
            layers = [nn.ConvTranspose2d(in_channels, out_channels, (3 if stride == 1 else 4), (stride, 2), 1)]
        else:
            layers = [nn.ConvTranspose2d(in_channels, out_channels, 3, padding=1)]
        if normalize:
            layers += [nn.InstanceNorm2d(out_channels)]
        layers += [nn.LeakyReLU(0.2, inplace=True)]
        if dropout:
            layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x.view(-1, 128, 2, 1))
        return x

class Conv2dEncoder_v2(torch.nn.Module):
    def __init__(self):
        super(Conv2dEncoder_v2, self).__init__()

        self.model = nn.Sequential(
            self.conv_layer(2, 4, stride=1, normalize=False),
            self.conv_layer(4, 8, down=False),
            self.conv_layer(8, 12, down=False),
            self.conv_layer(12, 16, down=False),
            self.conv_layer(16, 24),
            self.conv_layer(24, 32),
            self.conv_layer(32, 48, down=False),
            self.conv_layer(48, 64, dropout=0.5),
            self.conv_layer(64, 72, dropout=0.5),
            self.conv_layer(72, 96, dropout=0.5),
            self.conv_layer(96, 108, dropout=0.5),
            self.conv_layer(108, 128, dropout=0.5, normalize=False),
        )

    def conv_layer(self, in_channel, out_channel, stride=2, normalize=True, dropout=0.0, down=True):
        if down:
            layers = [nn.Conv2d(in_channel, out_channel, (3 if stride == 1 else 4, 4), (stride, 2), (1, 1), bias=False)]
        else:
            layers = [nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channel))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x.flatten(start_dim=1)

class Conv2dDecoder_v2(torch.nn.Module):
    def __init__(self):
        super(Conv2dDecoder_v2, self).__init__()
        self.model = nn.Sequential(
            self.conv_layer(128, 108, dropout=0.5, normalize=False),
            self.conv_layer(108, 96, dropout=0.5),
            self.conv_layer(96, 72, dropout=0.5),
            self.conv_layer(72, 64, dropout=0.5),
            self.conv_layer(64, 48, dropout=0.5),
            self.conv_layer(48, 32, up=False),
            self.conv_layer(32, 24, dropout=0.5),
            self.conv_layer(24, 16, dropout=0.5),
            self.conv_layer(16, 12, up=False),
            self.conv_layer(12, 8, up=False),
            self.conv_layer(8, 4, up=False),
            nn.ConvTranspose2d(4, 2, (3, 4), (1, 2), 1)
        )

    def conv_layer(self, in_channels, out_channels, stride=2, dropout=0.0, normalize=True, up=True):
        if up:
            layers = [nn.ConvTranspose2d(in_channels, out_channels, (3 if stride == 1 else 4), (stride, 2), 1)]
        else:
            layers = [nn.ConvTranspose2d(in_channels, out_channels, 3, padding=1)]
        if normalize:
            layers += [nn.InstanceNorm2d(out_channels)]
        layers += [nn.LeakyReLU(0.2, inplace=True)]
        if dropout:
            layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x.view(-1, 128, 1, 2))
        return x

class Discriminator1x2(torch.nn.Module):
    def __init__(self):
        super(Discriminator1x2, self).__init__()
        self.model = nn.Sequential(
            self.discriminator_layer(12*2 + 12*2, 48, stride=1, normalization=False),
            self.discriminator_layer(48, 56, stride=1),
            self.discriminator_layer(56, 64),
            self.discriminator_layer(64, 64),
            self.discriminator_layer(64, 72),
            self.discriminator_layer(72, 84),
            self.discriminator_layer(84, 96),
        )
        self.out = torch.nn.Conv2d(96, 1, (1, 1))

    def discriminator_layer(self, in_channels, out_channels, stride=2, normalization=True):
        layers = [nn.Conv2d(in_channels, out_channels, 4, stride=(stride, 2), padding=1)]
        if normalization:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x, x_r):
        x = self.model(torch.cat([x, x_r], dim=1))
        x = self.out(F.relu(x))
        return x.flatten(start_dim=1)

class Discriminator_soft2d(torch.nn.Module):
    def __init__(self):
        super(Discriminator_soft2d, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=(3, 3)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(24, 24, kernel_size=(3, 3), stride=(2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(24),
            nn.Dropout(0.5),
            nn.Conv2d(24, 32, kernel_size=(3, 3)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(32),
            nn.Dropout(0.5),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(32),
            nn.Dropout(0.5),
            nn.Conv2d(32, 64, kernel_size=(3, 3)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(64),
            nn.Dropout(0.5),
            nn.Conv2d(64, 72, kernel_size=(3, 3)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(72),
            nn.Dropout(0.5),
            nn.Conv2d(72, 84, kernel_size=(3, 3)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(64),
            nn.Dropout(0.5),
            nn.Conv2d(84, 108, kernel_size=(3, 3), stride=(2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(72),
            nn.Dropout(0.5),
            nn.Conv2d(108, 108, kernel_size=(3, 3)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(108),
            nn.Dropout(0.5),
            nn.Conv2d(108, 128, kernel_size=(3, 3),stride=(2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(128),
            nn.Dropout(0.5),
            nn.Conv2d(128, 128, kernel_size=(3, 3)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(128),
            nn.Dropout(0.5),
            nn.Flatten(start_dim=1),
            nn.Linear(128*2*2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder_Soft2d_2048(torch.nn.Module):
    def __init__(self, z=128):
        super(Decoder_Soft2d_2048, self).__init__()
        self.z = z
        self.generator = torch.nn.Sequential(
            nn.ConvTranspose2d(128, 108, kernel_size=(3, 3), padding=(1, 1)),
            nn.InstanceNorm2d(108),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(108, 96, kernel_size=(3, 4), padding=(1, 1), stride=(1, 2)),
            nn.InstanceNorm2d(96),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(96, 84, kernel_size=(3, 3), padding=(1, 1)),
            nn.InstanceNorm2d(84),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(84, 72, kernel_size=(3, 3), padding=(1, 1)),
            nn.InstanceNorm2d(73),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(72, 72, kernel_size=(4, 4), padding=(1, 1), stride=(2, 2)),
            nn.InstanceNorm2d(73),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(72, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(64, 48, kernel_size=(4, 4), padding=(1, 1), stride=(2, 2)),
            nn.InstanceNorm2d(48),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(48, 36, kernel_size=(3, 3), padding=(1, 1)),
            nn.InstanceNorm2d(36),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(36, 36, kernel_size=(4, 4), padding=(1, 1), stride=(2, 2)),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(36, 36, kernel_size=(3, 3), padding=(1, 1)),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(36, 32, kernel_size=(4, 4), padding=(1, 1), stride=(2, 2)),
            nn.InstanceNorm2d(24),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 28, kernel_size=(3, 3), padding=(1, 1)),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(28, 28, kernel_size=(4, 4), padding=(1, 1), stride=(2, 2)),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(28, 24, kernel_size=(3, 3), padding=(1, 1)),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(24, 24, kernel_size=(3, 3), padding=(1, 1)),
        )

    def forward(self, x):
        x = self.generator(x.view(-1, 128, 4, 2))
        return x

class Encoder_Soft2d_2048(torch.nn.Module):
    def __init__(self):
        super(Encoder_Soft2d_2048, self).__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=(3, 3), padding=1),
            nn.InstanceNorm2d(24),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Conv2d(24, 32, kernel_size=(4, 4), padding=(1, 1), stride=(2, 2)),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Conv2d(32, 48, kernel_size=(4, 4), padding=1, stride=(2, 2)),
            nn.InstanceNorm2d(48),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Conv2d(48, 64, kernel_size=(4, 4), padding=(1, 1), stride=(2, 2)),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Conv2d(64, 64, kernel_size=(4, 4), padding=1, stride=(2, 2)),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Conv2d(64, 72, kernel_size=(4, 4), padding=(1, 1), stride=(2, 2)),
            nn.InstanceNorm2d(72),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Conv2d(72, 72, kernel_size=(3, 3), padding=1),
            nn.InstanceNorm2d(72),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Conv2d(72, 84, kernel_size=(3, 4), padding=(1, 1), stride=(1, 2)),
            nn.InstanceNorm2d(84),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Conv2d(84, 96, kernel_size=(3, 3), padding=1),
            nn.InstanceNorm2d(96),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Conv2d(96, 128, kernel_size=(3, 3), padding=1),
        )

    def forward(self, x):
        return self.encode(x).flatten(start_dim=1)
        
def MMD(x, y, kernel='rbf'):
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz  # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).cuda(),
                  torch.zeros(xx.shape).cuda(),
                  torch.zeros(xx.shape).cuda())

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a ** 2 * (a ** 2 + dxx) ** -1
            YY += a ** 2 * (a ** 2 + dyy) ** -1
            XY += a ** 2 * (a ** 2 + dxy) ** -1

    if kernel == "rbf":
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx.cuda() / a)
            YY += torch.exp(-0.5 * dyy.cuda() / a)
            XY += torch.exp(-0.5 * dxy.cuda() / a)

    return torch.mean(XX + YY - 2. * XY)


def evaluate_l1_loss(y_hat, y, ref, alpha=0.5):
    ref[ref == 1.0] = alpha
    ref[ref == 0.0] = 1.0 - alpha
    loss = F.l1_loss(y, y_hat.squeeze(dim=1), reduction='none') * ref
    return loss.mean(dim=-1).mean() * 10

def transform(m, a):
    m, a = m[:, 14:114], a[:, 14:114]
    inv_spec = torchaudio.transforms.InverseSpectrogram(win_length=6, n_fft=255)
    m, a = m.unsqueeze(dim=0), a.unsqueeze(dim=0)
    z = torch.complex(m * torch.cos(a) , m * torch.sin(a))
    return inv_spec(z.cpu()).squeeze(dim=0)

# data loader
import pickle

recording_to_files = None
with open(r"./Data/D3/subject_to_recording.pickle", "rb") as input_file:
    recording_to_files = pickle.load(input_file)

from sklearn.model_selection import train_test_split

files_train, files_test = train_test_split(np.array(list(recording_to_files.keys())), test_size=0.1, random_state=42)

X_train = []
for file in files_train:
    if file in recording_to_files:
        X_train += recording_to_files[file]
X_test = []
for file in files_test:
    if file in recording_to_files:
        X_test += recording_to_files[file]

X_train = np.array(X_train)
X_test = np.array(X_test)
print('train on: {:d}, validate on {:d}'.format(X_train.shape[0], X_test.shape[0]))


params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 8}

params_te = {'batch_size': 64,
             'shuffle': False,
             'num_workers': 8}

training_set = Dataset1x2_Rec(X_train)
training_generator = torch.utils.data.DataLoader(training_set, **params)

validation_set = Dataset1x2_Rec(X_test)
validation_generator = torch.utils.data.DataLoader(validation_set, **params_te)

Z = 72
encoder = Encoder_Soft2d_2048()
decoder = Decoder_Soft2d_2048()
discriminator = Discriminator_soft2d()

encoder, decoder, discriminator = encoder.cuda(), decoder.cuda(), discriminator.cuda()

enc_optim = torch.optim.Adam(encoder.parameters(), lr=0.001)
dec_optim = torch.optim.Adam(decoder.parameters(), lr=0.001)
dis_optim = torch.optim.Adam(discriminator.parameters(), lr=0.001)

# dis_optim = torch.optim.Adam(discriminator .parameters(), lr=0.00001)

print('# of parameters (enc):', sum(p.numel() for p in encoder.parameters() if p.requires_grad))
print('# of parameters (dec):', sum(p.numel() for p in decoder.parameters() if p.requires_grad))
print('# of parameters (dec):', sum(p.numel() for p in discriminator.parameters() if p.requires_grad))

DIR = 'T4/V1'
code = 'waeganpix-{}'.format(DIR.replace('/', '-'))

# load and evaluate -------
dirname = './SIN/{}/f{:d}'.format(DIR, 100)

encoder.load_state_dict(torch.load('{}/enc_{}.tm'.format(dirname, code)))
decoder.load_state_dict(torch.load('{}/dec_{}.tm'.format(dirname, code)))

min_loss = -np.inf
last_improvmnt = -1

epoch_train, epoch_test = [], []
gener_loss, discrim_loss = [], []

train_dis = False

for epoch in range(1, 1000):
    print('---Epoch---{:d}-{}'.format(epoch, code))
    encoder.train()
    decoder.train()

    train_loss, train_acc, g_loss, d_loss = [], [], [], []

    for i, (x, ref) in enumerate(training_generator):
        enc_optim.zero_grad()
        dec_optim.zero_grad()

        valid = Variable(torch.from_numpy(np.ones((x.shape[0], 1))), requires_grad=False).float().cuda()
        fake = Variable(torch.from_numpy(np.zeros((x.shape[0], 1))), requires_grad=False).float().cuda()

        # train-the-generator
        z = encoder(x.cuda())
        x_recons = decoder(z)

        # compute the MAD loss
        recons_loss = F.mse_loss(x_recons, ref.cuda())

        # MMD loss
        z_fake = torch.autograd.Variable(torch.randn(z.shape[0], z.shape[-1]) * 1.0).cuda()
        z_real = encoder(x.cuda())
        mmd_loss = MMD(z_real, z_fake, kernel='rbf')

        # GAN loss
        loss_gan = F.mse_loss(discriminator(x_recons), valid)

        # condition
        if epoch % 3 == 0 and epoch > 20 and loss_gan.item() > 0.20:  # with even
            train_dis = True
        else:
            train_dis = False

        if train_dis:
            loss = recons_loss + 10 * mmd_loss + loss_gan
        else:
            loss = recons_loss + 10 * mmd_loss

        loss.backward(retain_graph=True)
        enc_optim.step()
        dec_optim.step()

        # train-the-discriminator
        real_loss = F.mse_loss(discriminator(ref.cuda()), valid)
        fake_loss = F.mse_loss(discriminator(x_recons.detach()), fake)

        loss_disc = 0.5 * (real_loss + fake_loss)

        if train_dis:
            dis_optim.zero_grad()
            loss_disc.backward(retain_graph=True)
            dis_optim.step()

        g_loss.append(loss_gan.item())
        d_loss.append(loss_disc.item())

        print('\r[{:4d}]train loss: {:.4f} recons: {:.4f} mmd: {:.4f} |G: {} GAN (G): {:.4f} GAN (D): {:.4f}'.format(i,
                                                                                                                     loss.item(),
                                                                                                                     recons_loss.item(),
                                                                                                                     mmd_loss.item(),
                                                                                                                     train_dis,
                                                                                                                     loss_gan.item(),
                                                                                                                     loss_disc.item()),
              end='')
        train_loss.append(recons_loss.item())

        #if i == 10:
        #    break

    print('\n---({}) Train Loss: {:.4f} GAN (G): {:.4f} GAN (D): {:.4f}'.format(code, np.mean(train_loss),
                                                                                np.mean(g_loss), np.mean(d_loss)))
    epoch_train.append(np.mean(train_loss))
    # save-render

    gener_loss.append(np.mean(g_loss))
    discrim_loss.append(np.mean(d_loss))

    _ = plt.figure(figsize=(12, 5))

    plt.subplot(311)
    plt.plot(gener_loss)
    plt.title('generator')

    plt.subplot(312)
    plt.plot(discrim_loss)
    plt.title('discriminator ')

    plt.subplot(313)
    plt.plot(epoch_train)
    plt.title('recons-L2')

    plt.tight_layout()
    plt.savefig('./Loss/loss-{}.png'.format(code))
    plt.clf()
    plt.close()

    encoder.eval()
    decoder.eval()
    test_recons, test_acc = [], []
    for i, (x, s) in enumerate(validation_generator):

        # train-the-generator
        z = encoder(x.cuda())
        x_recons = decoder(z)

        loss_pixel = F.l1_loss(x_recons, s.cuda())
        test_recons.append(loss_pixel.item())

        print('\r[{:4d}]test pixel: {:.4f}'.format(i, loss_pixel.item()), end='')

        if i == 10:
            ids = np.random.choice(range(x.shape[0]), 100, replace=True)    
            dirname = './SIN/{}/f{:d}'.format(DIR, epoch)
            import os 
            try:
                os.mkdir(dirname)
            except FileExistsError:
                pass

            for k in range(100):
                _ = plt.figure(figsize=(15, 6))
                
                plt.subplot(5,3,1)
                plt.plot(transform(x[ids[k],0,:], x[ids[k], 0+12, :]).detach().cpu().numpy())

                for l in range(12):
                    plt.subplot(5,3,l+1 + 3)

                    # plot the real signal
                    x_r = transform(s[ids[k], l, ], s[ids[k], l+12, ])
                    plt.plot(x_r.detach().cpu().numpy(), 'g', alpha=0.4)

                    # plot the reconstructed signal
                    x_re = transform(x_recons[ids[k], l, ], x_recons[ids[k], l+12, ])
                    plt.plot(x_re.detach().cpu().numpy())

                plt.tight_layout()
                plt.savefig('{}/f_{:d}.png'.format(dirname, k))
                plt.close()
                plt.clf()

            #break
            
            
    print('\n------({}) Test recons: {:.4f}'.format(code, np.mean(test_recons)))

    # save models
    torch.save(encoder.state_dict(), '{}/enc_{}.tm'.format(dirname, code))
    torch.save(decoder.state_dict(), '{}/dec_{}.tm'.format(dirname, code))
    torch.save(discriminator.state_dict(), '{}/dis_{}.tm'.format(dirname, code))

    torch.save(enc_optim.state_dict(), '{}/enc_opt_{}.tm'.format(dirname, code))
    torch.save(dec_optim.state_dict(), '{}/dec_opt_{}.tm'.format(dirname, code))
    torch.save(dis_optim.state_dict(), '{}/dis_opt_{}.tm'.format(dirname, code))
