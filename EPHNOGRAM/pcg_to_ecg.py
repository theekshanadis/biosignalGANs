import h5py
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

class Dataset1x2_Rec(torch.utils.data.Dataset):
    def __init__(self, indexes, test=False):
        self.indexes = indexes
        self.stft_f = torchaudio.transforms.Spectrogram(win_length=16, n_fft=255, power=None)

    def create_features(self, x):
        x_spec = self.stft_f(torch.from_numpy(x).unsqueeze(dim=0).float())
        spec_mag, spec_ang = x_spec.abs(), x_spec.angle()

        spec_mag = F.pad(spec_mag, pad=(6, 6), mode='constant', value=0)
        spec_ang = F.pad(spec_ang, pad=(6, 6), mode='constant', value=0)

        return spec_mag, spec_ang

    def __getitem__(self, index):
        x = np.load(self.indexes[index])

        x_spec = self.stft_f(torch.from_numpy(x).float())
        spec_mag, spec_ang = x_spec.abs(), x_spec.angle()

        spec_mag = F.pad(spec_mag, pad=(3, 3), mode='constant', value=0)
        spec_ang = F.pad(spec_ang, pad=(3, 3), mode='constant', value=0)

        x = torch.cat([spec_mag, spec_ang], dim=0)

        ecg, pcg = x[[0, 2], ], x[[1, 3], ]

        return ecg, pcg

    def __len__(self):
        return self.indexes.shape[0]

class Dataset1x2_Rec_(torch.utils.data.Dataset):
    def __init__(self, indexes, s, test=False):
        self.indexes = indexes
        self.subjacts = s
        self.stft_f = torchaudio.transforms.Spectrogram(win_length=16, n_fft=255, power=None)

    def create_features(self, x):
        x_spec = self.stft_f(torch.from_numpy(x).unsqueeze(dim=0).float())
        spec_mag, spec_ang = x_spec.abs(), x_spec.angle()

        spec_mag = F.pad(spec_mag, pad=(6, 6), mode='constant', value=0)
        spec_ang = F.pad(spec_ang, pad=(6, 6), mode='constant', value=0)

        return spec_mag, spec_ang

    def __getitem__(self, index):
        x = np.load(self.indexes[index])

        x_spec = self.stft_f(torch.from_numpy(x).float())
        spec_mag, spec_ang = x_spec.abs(), x_spec.angle()

        spec_mag = F.pad(spec_mag, pad=(3, 3), mode='constant', value=0)
        spec_ang = F.pad(spec_ang, pad=(3, 3), mode='constant', value=0)

        x = torch.cat([spec_mag, spec_ang], dim=0)

        ecg, pcg = x[[0, 2], ], x[[1, 3], ]

        return ecg, pcg, self.subjacts[index]

    def __len__(self):
        return self.indexes.shape[0]

class UnetDown1x2(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, normalize=True, dropout=0.0, prop=False):
        super(UnetDown1x2, self).__init__()
        if prop:
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
        else:
            layers = [nn.Conv2d(in_channels, out_channels, (3 if stride ==1 else 4, 4), (stride, 2), (1, 1), bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append((nn.Dropout(dropout)))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UnetUp1x2(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, dropout=0.0, normalize=True, prop=False):
        super(UnetUp1x2, self).__init__()
        if prop:
            layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1)]
        else:
            layers = [nn.ConvTranspose2d(in_channels, out_channels, (3 if stride == 1 else 4), (stride, 2), 1)]
        if normalize:
            layers += [nn.InstanceNorm2d(out_channels)]
        layers += [nn.LeakyReLU(0.2, inplace=True)]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, x_sk):
        x = self.model(x)
        x = torch.cat([x, x_sk], 1)
        return x

class Discriminator1x2(torch.nn.Module):
    def __init__(self):
        super(Discriminator1x2, self).__init__()
        self.discriminator = torch.nn.Sequential(
            nn.Conv2d(4, 12, kernel_size=(3, 3)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(12),
            nn.Dropout(0.4),
            nn.Conv2d(12, 24, kernel_size=(3, 3), stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(24),
            nn.Dropout(0.4),
            nn.Conv2d(24, 32, kernel_size=(3, 3)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(32),
            nn.Dropout(0.4),
            nn.Conv2d(32, 48, kernel_size=(3, 3), stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(48),
            nn.Dropout(0.4),
            nn.Conv2d(48, 64, kernel_size=(3, 3)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(64),
            nn.Dropout(0.4),
            nn.Conv2d(64, 84, kernel_size=(3, 3), stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(84),
            nn.Dropout(0.4),
            nn.Conv2d(84, 96, kernel_size=(3, 3)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(96),
            nn.Dropout(0.4),
            nn.Conv2d(96, 108, kernel_size=(3, 3), stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(108),
            nn.Dropout(0.4),
            nn.Conv2d(108, 128, kernel_size=(3, 3)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(128),
            nn.Dropout(0.4),
            nn.Conv2d(128, 144, kernel_size=(3, 3), stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(12),
            nn.Dropout(0.4),
            nn.Conv2d(144, 1, kernel_size=(1, 1))
        )

    def forward(self, x, x_sk):
        x = self.discriminator(torch.cat([x, x_sk], dim=1)).flatten(start_dim=1)
        return x

class GeneraterUNet1x2_v4(torch.nn.Module):
    def __init__(self):
        super(GeneraterUNet1x2_v4, self).__init__()

        self.down0 = UnetDown1x2(2, 8, normalize=False, prop=True)
        self.down1 = UnetDown1x2(8, 12)
        self.down2 = UnetDown1x2(12, 24)
        self.down3 = UnetDown1x2(24, 32)
        self.down4 = UnetDown1x2(32, 48, dropout=0.5, prop=True)
        self.down5 = UnetDown1x2(48, 48, dropout=0.5, prop=True)
        self.down6 = UnetDown1x2(48, 64, dropout=0.5)
        self.down7 = UnetDown1x2(64, 72, dropout=0.5, prop=True)
        self.down8 = UnetDown1x2(72, 84, dropout=0.5, prop=True)
        self.down9 = UnetDown1x2(84, 96, dropout=0.5)

        self.up2 = UnetUp1x2(96, 84, dropout=0.5, normalize=False)
        self.up3 = UnetUp1x2(84 * 2, 72, dropout=0.5, prop=True)
        self.up4 = UnetUp1x2(72 * 2, 64, dropout=0.5, prop=True)
        self.up5 = UnetUp1x2(64 * 2, 48, dropout=0.5)
        self.up6 = UnetUp1x2(48 * 2, 48, dropout=0.5, prop=True)
        self.up7 = UnetUp1x2(48 * 2, 32, dropout=0.5, prop=True)
        self.up8 = UnetUp1x2(32 * 2, 24, dropout=0.5)
        self.up9 = UnetUp1x2(24 * 2, 12)
        self.up10 = UnetUp1x2(12 * 2, 8)

        self.final = torch.nn.Sequential(
            nn.ConvTranspose2d(8*2, 2, (3, 3), (1, 1), 1),
        )

    def forward(self, x):
        d0 = self.down0(x)
        d1 = self.down1(d0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        z = self.down9(d8)

        u2 = self.up2(z, d8)
        u3 = self.up3(u2, d7)
        u4 = self.up4(u3, d6)
        u5 = self.up5(u4, d5)
        u6 = self.up6(u5, d4)
        u7 = self.up7(u6, d3)
        u8 = self.up8(u7, d2)
        u9 = self.up9(u8, d1)
        u10 = self.up10(u9, d0)

        out = self.final(u10)
        return out, z.flatten(start_dim=1)

def transform(x):
    m, a = x[:, 0,:, 3:253], x[:, 1, :, 3:253]
    inv_spec = torchaudio.transforms.InverseSpectrogram(n_fft=255, win_length=16)
    z = torch.complex(m * torch.cos(a) , m * torch.sin(a))
    return inv_spec(z.cpu())

# data loader
import pickle
recording_to_files = None
with open(r"/media/PPGECG/Data4/subject_to_recording.pickle", "rb") as input_file:
    recording_to_files = pickle.load(input_file)
from sklearn.model_selection import train_test_split

import pandas as pd 
df = pd.read_csv('ECGPCGSpreadsheet.csv')
X = []

sub_to_id = {}
subjects = ['S023', 'S003', 'S004', 'S005', 'S006', 'S007', 'S008', 'S009', 'S001', 'S010', 'S002', 'S012', 'S013', 'S014', 'S015', 'S016', 'S017', 'S018', 'S019', 'S011', 'S020']
for i in range(len(subjects)):
    sub_to_id[subjects[i]] = i

s = []
for record, condition, sid in zip(df['Record Name'], df['ECG Notes'], df['Subject ID']):
    if condition == 'Good':
        X += recording_to_files['{}/{}'.format('WFDB', record)]
        s += [sub_to_id[sid]] * len(recording_to_files['{}/{}'.format('WFDB', record)])

from sklearn.model_selection import train_test_split
X_train, X_test, _, y_test = train_test_split(np.array(X), np.array(s), test_size=0.1, random_state=42)

print('train on: {:d}, validate on {:d}'.format(X_train.shape[0], X_test.shape[0]))

params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6}

params_te = {'batch_size': 64,
          'shuffle': False,
          'num_workers': 6}

training_set = Dataset1x2_Rec(X_train)
training_generator = torch.utils.data.DataLoader(training_set, **params)

validation_set = Dataset1x2_Rec_(X_test, y_test)
validation_generator = torch.utils.data.DataLoader(validation_set, **params_te)

generator = GeneraterUNet1x2_v4()
discriminator  = Discriminator1x2()

criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()

generator, discriminator  = generator.cuda(), discriminator.cuda()

enc_optim = torch.optim.Adam(generator.parameters(), lr=0.001)
dis_optim = torch.optim.Adam(discriminator .parameters(), lr=0.001)

#dis_optim = torch.optim.Adam(discriminator .parameters(), lr=0.00001)

print('# of parameters (enc):', sum(p.numel() for p in generator.parameters() if p.requires_grad))
print('# of parameters (dec):', sum(p.numel() for p in discriminator .parameters() if p.requires_grad))

DIR = 'T7/V1'

code = 'pixtopix_ecgppg_{}'.format(DIR.replace('/', '-'))
dirname = './SIN/{}/f{:d}'.format(DIR, 100)

generator.load_state_dict(torch.load('{}/enc_{}.tm'.format(dirname, code)))

'''
####-------------plots
for i, (ecg, vsg) in enumerate(validation_generator):
    # compute the MAD loss
    x_recons = generator(vsg.cuda())
    loss_pixel = criterion_pixelwise(x_recons, ecg.cuda())
    print('\r[{:4d}]test pixel: {:.4f}'.format(i, loss_pixel.item()), end='')

    if i in [1, 100, 500, 750, 1000]:
        dirname = './Final/V1'
        import os 
        try:
            os.mkdir(dirname)
        except FileExistsError:
            pass

        x_recons = transform(x_recons)
        ecg = transform(ecg)
        vsg = transform(vsg)

        for k in range(ecg.shape[0]):
            _ = plt.figure(figsize=(5, 5))

            plt.subplot(311)
            plt.plot(vsg[k, ].detach().cpu().numpy())
            plt.title('$x$')

            plt.subplot(312)
            plt.plot(ecg[k, ].detach().cpu().numpy())
            plt.title('$y$')

            plt.subplot(313)
            plt.plot(x_recons[k, ].detach().cpu().numpy(), 'g')
            plt.title('$\hat{y}$')

            plt.tight_layout()
            plt.savefig('{}/f_{:d}_{:d}.png'.format(dirname, k, i))
            plt.close()
            plt.clf()

        torch.save(x_recons.detach().cpu(), '{}/x_recons_{:d}.tm'.format(dirname, i))
        torch.save(ecg.detach().cpu(), '{}/y_{:d}.tm'.format(dirname, i))
        torch.save(vsg.detach().cpu(), '{}/x_{:d}.tm'.format(dirname, i))
'''

# t-SNE
# on validation data
generator.eval()

z_ = []
c_ = []
for i, (x, s, y) in enumerate(validation_generator):
    # train-the-generator
    _, z = generator(x.cuda())
    print(i)
    if i == 64:
        break

    z_.append(z.cpu())
    c_.append(y)

z = torch.cat(z_, dim=0)
c = torch.cat(c_, dim=0)

# tSNE
from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2, verbose=1, perplexity=33, n_iter=10000, learning_rate=100).fit_transform(z.detach().numpy())

plt.style.use('ggplot')
for i in range(21):
    x_ = X_embedded[c == i, ]
    plt.scatter(x_[:, 0], x_[:, 1], s=8, label='class-{}'.format(i))

#plt.legend()
plt.savefig('pcg_ecg.png')
exit()

min_loss = -np.inf
last_improvmnt = -1
gener_loss, discrim_loss, pixel_loss = [], [], []

train_dis = False

for epoch in range(1000):
    print('---Epoch---{:d}-{}'.format(epoch, code))
    generator.train()
    train_loss, train_acc, g_loss, d_loss, p_loss = [], [], [], [], []

    for i, (ecg, vsg) in enumerate(training_generator):        
        # adv-ground truths
        shape = (ecg.shape[0], 5)
        valid = Variable(torch.from_numpy(np.ones(shape)), requires_grad=False).float().cuda()
        fake = Variable(torch.from_numpy(np.zeros(shape)), requires_grad=False).float().cuda()

        # -----------------G--------------------
        enc_optim.zero_grad()
        # gan_loss
        fake_B = generator(vsg.cuda())
        pred_fake = discriminator (vsg.cuda(), fake_B.cuda())
        loss_gan = criterion_GAN(pred_fake, valid)

        # pixel_loss
        loss_pixel = criterion_pixelwise(fake_B, ecg.cuda())

        # total loss
        loss_G = loss_gan + loss_pixel
        loss_G.backward()
        enc_optim.step()

        # ---------------D------------------
        pred_real = discriminator (vsg.cuda(), ecg.cuda())
        loss_real = criterion_GAN(pred_real, valid)

        pred_fake = discriminator (vsg.cuda(), fake_B.detach())
        loss_fake = criterion_GAN(pred_fake, fake)

        loss_D = 0.5 * (loss_real + loss_fake)

        if epoch % 4 != 0:
            dis_optim.zero_grad()
            loss_D.backward()
            dis_optim.step()

        g_loss.append(loss_G.item())
        d_loss.append(loss_D.item())
        p_loss.append(loss_pixel.item())

        print('\r[{:4d}]train (G) pixel: {:.4f} loss_G ({}): {:.4f}, (D): {:.4f}'.format(i, loss_pixel.item(), epoch % 4 != 0, loss_G.item(), loss_D.item()), end='')

        #if i == 2:
        #    break

    print('\n---({}) Train GAN (G): {:.4f} GAN (D): {:.4f}'.format(code, np.mean(g_loss), np.mean(d_loss))) 
    # save-render
    gener_loss.append(np.mean(g_loss))
    discrim_loss.append(np.mean(d_loss))
    pixel_loss.append(np.mean(p_loss))

    _ = plt.figure(figsize=(12, 5))
    
    plt.subplot(311)
    plt.plot(gener_loss)
    plt.title('generator')

    plt.subplot(312)
    plt.plot(discrim_loss)
    plt.title('discriminator ')

    plt.subplot(313)
    plt.plot(pixel_loss)
    plt.title('recons-L1')

    plt.tight_layout()
    plt.savefig('./Loss/loss-{}.png'.format(code))
    plt.clf()
    plt.close()

    generator.eval()
    test_recons, test_acc = [], []
    for i, (ecg, vsg) in enumerate(validation_generator):
        
        # compute the MAD loss
        x_recons = generator(vsg.cuda())

        loss_pixel = criterion_pixelwise(x_recons, ecg.cuda())

        test_recons.append(loss_pixel.item())

        print('\r[{:4d}]test pixel: {:.4f}'.format(i, loss_pixel.item()), end='')

        if i == 1:

            dirname = './SIN/{}/f{:d}'.format(DIR, epoch)
            import os 
            try:
                os.mkdir(dirname)
            except FileExistsError:
                pass

            x_recons = transform(x_recons)
            ecg = transform(ecg)
            vsg = transform(vsg)

            for k in range(ecg.shape[0]):
                _ = plt.figure(figsize=(5, 5))

                plt.subplot(311)
                plt.plot(vsg[k, ].detach().cpu().numpy())
                plt.title('$x$')

                plt.subplot(312)
                plt.plot(ecg[k, ].detach().cpu().numpy())
                plt.title('$y$')

                plt.subplot(313)
                plt.plot(x_recons[k, ].detach().cpu().numpy(), 'g')
                plt.title('$\hat{y}$')

                plt.tight_layout()
                plt.savefig('{}/f_{:d}.png'.format(dirname, k))
                plt.close()
                plt.clf()
            #break

    print('\n------({}) Test recons: {:.4f}'.format(code, np.mean(test_recons)))

    # save models
    torch.save(generator.state_dict(), '{}/enc_{}.tm'.format(dirname, code))
    torch.save(discriminator .state_dict(), '{}/dec_{}.tm'.format(dirname, code))

    torch.save(enc_optim.state_dict(), '{}/enc_opt_{}.tm'.format(dirname, code))
    torch.save(dis_optim.state_dict(), '{}/dec_opt_{}.tm'.format(dirname, code))



