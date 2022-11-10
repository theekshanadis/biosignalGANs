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

        ecg, pcg = x[[0, 2],], x[[1, 3],]

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

        ecg, pcg = x[[0, 2],], x[[1, 3],]

        return ecg, pcg, self.subjacts[index]

    def __len__(self):
        return self.indexes.shape[0]


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
            self.conv_layer(72, 84, dropout=0.5),
            nn.Conv2d(84, 96, 4, 2, 1),
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
        z = self.model(x)
        return z

class Conv2dDecoder_v2(torch.nn.Module):
    def __init__(self):
        super(Conv2dDecoder_v2, self).__init__()
        self.model = nn.Sequential(
            self.conv_layer(96, 84, dropout=0.5),
            self.conv_layer(84, 72, dropout=0.5),
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
        x = self.model(x.view(-1, 96, 2, 2))
        return x

class GeneratorNet(nn.Module):
    def __init__(self):
        super(GeneratorNet, self).__init__()
        self.encoder = Conv2dEncoder_v2()
        self.decoder = Conv2dDecoder_v2()
        self.activate = nn.LeakyReLU(0.4, inplace=True)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(self.activate(z))
        return x_hat

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2, 12, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Conv2d(12, 24, kernel_size=(4, 4), padding=1, stride=(2, 2)),
            nn.InstanceNorm2d(24),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Conv2d(24, 32, kernel_size=(4, 4), padding=1, stride=(2, 2)),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Conv2d(32, 32, kernel_size=(4, 4), padding=1, stride=(2, 2)),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Conv2d(32, 36, kernel_size=(4, 4), padding=1, stride=(2, 2)),
            nn.InstanceNorm2d(36),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Conv2d(36, 48, kernel_size=(4, 4), padding=1, stride=(2, 2)),
            nn.InstanceNorm2d(48),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Conv2d(48, 64, kernel_size=(4, 4), padding=1, stride=(2, 2)),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Conv2d(64, 72, kernel_size=(4, 4), padding=1, stride=(2, 2)),
            nn.InstanceNorm2d(72),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Flatten(start_dim=1),
            nn.Linear(144, 1),
        )

    def forward(self, x):
        return self.model(x)


def transform(x):
    m, a = x[:, 0, :, 3:253], x[:, 1, :, 3:253]
    inv_spec = torchaudio.transforms.InverseSpectrogram(n_fft=255, win_length=16)
    z = torch.complex(m * torch.cos(a), m * torch.sin(a))
    return inv_spec(z.cpu())


# data loader
import pickle5 as pickle

recording_to_files = None
with open(r"./Data/subject_to_recording.pickle", "rb") as input_file:
    recording_to_files = pickle.load(input_file)
from sklearn.model_selection import train_test_split

import pandas as pd

df = pd.read_csv('ECGPCGSpreadsheet_mod.csv')
X = []

sub_to_id = {}
subjects = ['S023', 'S003', 'S004', 'S005', 'S006', 'S007', 'S008', 'S009', 'S001', 'S010', 'S002', 'S012', 'S013',
            'S014', 'S015', 'S016', 'S017', 'S018', 'S019', 'S011', 'S020']
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

G_AB = GeneratorNet()
G_BA = GeneratorNet()

D_A = Discriminator()
D_B = Discriminator()

G_BA, G_AB, D_A, D_B = G_BA.cuda(), G_AB.cuda(), D_A.cuda(), D_B.cuda()

import itertools

optim_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=0.01)

optim_D_A = torch.optim.Adam(D_A.parameters(), lr=0.001)
optim_D_B = torch.optim.Adam(D_B.parameters(), lr=0.001)

criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

import random
class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


print('# of parameters (enc):', sum(p.numel() for p in G_AB.parameters() if p.requires_grad))
print('# of parameters (dec):', sum(p.numel() for p in D_A.parameters() if p.requires_grad))

DIR = 'T3/V2'

code = 'cyclegan-{}'.format(DIR.replace('/', '-'))
dirname = './SIN/{}/f{:d}'.format(DIR, 100)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

min_loss = -np.inf
last_improvmnt = -1
gener_loss, discrim_loss, pixel_loss = [], [], []

train_dis = False

for epoch in range(1, 1000):
    print('---Epoch---{:d}-{}'.format(epoch, code))
    train_loss, train_acc, g_loss, d_loss, p_loss = [], [], [], [], []

    for i, (real_A, real_B) in enumerate(training_generator):
        # adv-ground truths
        shape = (real_A.shape[0], 1)
        valid = Variable(torch.from_numpy(np.ones(shape)), requires_grad=False).float().cuda()
        fake = Variable(torch.from_numpy(np.zeros(shape)), requires_grad=False).float().cuda()

        # -----------------G--------------------
        G_AB.train()
        G_BA.train()

        real_A = real_A.to('cuda')
        real_B = real_B.to('cuda')

        optim_G.zero_grad()
        loss_id_A = criterion_identity(G_BA(real_A), real_A)
        loss_id_B = criterion_identity(G_AB(real_B), real_B)
        loss_identity = 0.5 * (loss_id_B + loss_id_A)

        fake_B = G_AB(real_A)
        loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
        fake_A = G_BA(real_B)
        loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
        loss_GAN = 0.5 * (loss_GAN_BA + loss_GAN_AB)

        recov_A = G_BA(fake_B)
        loss_cycle_A = criterion_cycle(recov_A, real_A)
        recov_B = G_AB(fake_A)
        loss_cycle_B = criterion_cycle(recov_B, real_B)
        loss_cycle = 0.5 * (loss_cycle_B + loss_cycle_A)

        # for stability of the model, we added an additional L1 loss which was not in the original implementation. 
        l1_loss = 0.5*(F.l1_loss(fake_A, real_A) + F.l1_loss(fake_B, real_B))

        loss_G = loss_GAN + 2 * loss_cycle + 2 * loss_identity + l1_loss
        loss_G.backward()
        optim_G.step()
        # ----------------D-----------------
        
        loss_real = criterion_GAN(D_A(real_A), valid)
        fake_A_ = fake_A_buffer.push_and_pop(fake_A)
        loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
        loss_D_A = 0.5 * (loss_real + loss_fake)
        
        if epoch > 20 and epoch % 3 == 0:
            optim_D_A.zero_grad()
            loss_D_A.backward()
            optim_D_A.step()

        loss_real = criterion_GAN(D_B(real_B), valid)
        fake_B_ = fake_B_buffer.push_and_pop(fake_B)
        loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
        loss_D_B = 0.5 * (loss_real + loss_fake)
        
        if epoch > 20 and epoch % 3 == 0:
            optim_D_B.zero_grad()
            loss_D_B.backward()
            optim_D_B.step()

        g_loss.append(loss_G.item())
        d_loss.append(loss_D_A.item())
        p_loss.append(loss_D_B.item())

        print('\r[{:4d}]train (G): {:.4f}[{:.4f},{:.4f},{:.4f}] l1:{:.4f} (D(A,B){}): {:.4f}|{:.4f}'.format(i,
                                                                                                                     loss_G.item(),
                                                                                                                     loss_identity.item(),
                                                                                                                     loss_cycle.item(),
                                                                                                                     loss_GAN.item(),
                                                                                                                     l1_loss.item(),
                                                                                                                     epoch > 20 and epoch % 3 == 0,
                                                                                                                     loss_D_A.item(),
                                                                                                                     loss_D_B.item()),
              end='')

        #if i == 2:
        #    break

    print('\n---({}) Train GAN (G): {:.4f} GAN (D): {:.4f}|{:.4f}'.format(code, np.mean(g_loss), np.mean(d_loss),
                                                                          np.mean(p_loss)))
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
    plt.title('discriminatorA ')

    plt.subplot(313)
    plt.plot(pixel_loss)
    plt.title('discriminatorB')

    plt.tight_layout()
    plt.savefig('./Loss/loss-{}.png'.format(code))
    plt.clf()
    plt.close()

    G_AB.eval()
    G_BA.eval()
    test_recons, test_acc = [], []
    for i, (real_A, real_B, _) in enumerate(validation_generator):
        
        real_A = real_A.to('cuda')
        real_B = real_B.to('cuda')

        shape = (real_A.shape[0], 1)
        valid = Variable(torch.from_numpy(np.ones(shape)), requires_grad=False).float().cuda()
        fake = Variable(torch.from_numpy(np.zeros(shape)), requires_grad=False).float().cuda()

        loss_id_A = criterion_identity(G_BA(real_A), real_A)
        loss_id_B = criterion_identity(G_AB(real_B), real_B)
        loss_identity = 0.5 * (loss_id_B + loss_id_A)

        fake_B = G_AB(real_A)
        loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
        fake_A = G_BA(real_B)
        loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
        loss_GAN = 0.5 * (loss_GAN_BA + loss_GAN_AB)

        recov_A = G_BA(fake_B)
        loss_cycle_A = criterion_cycle(recov_A, real_A)
        recov_B = G_AB(fake_A)
        loss_cycle_B = criterion_cycle(recov_B, real_B)
        loss_cycle = 0.5 * (loss_cycle_B + loss_cycle_A)

        test_recons.append(loss_cycle.item())

        print('\r[{:4d}]test pixel: {:.4f}|{:.4f},{:.4f}'.format(i, loss_identity.item(), loss_GAN.item(), loss_cycle.item()), end='')
        if i == 1:
            dirname = './SIN/{}/f{:d}'.format(DIR, epoch)
            import os
            try:
                os.mkdir(dirname)
            except FileExistsError:
                pass
            fake_B = transform(fake_B)
            fake_A = transform(fake_A)
            
            real_A = transform(real_A)
            real_B = transform(real_B)
            
            for k in range(real_A.shape[0]):
                _ = plt.figure(figsize=(12, 3))

                plt.subplot(121)
                plt.plot(real_A[k,].detach().cpu().numpy())
                plt.plot(fake_A[k,].detach().cpu().numpy())
                plt.title('$a$')

                plt.subplot(122)
                plt.plot(real_B[k,].detach().cpu().numpy())
                plt.plot(fake_B[k,].detach().cpu().numpy())
                plt.title('$b$')

                plt.tight_layout()
                plt.savefig('{}/f_{:d}.png'.format(dirname, k))
                plt.close()
                plt.clf()
            #break

    print('\n------({}) Test recons: {:.4f}'.format(code, np.mean(test_recons)))

    # save models
    torch.save(G_AB.state_dict(), '{}/encab_{}.tm'.format(dirname, code))
    torch.save(G_BA.state_dict(), '{}/encba_{}.tm'.format(dirname, code))
    torch.save(D_B.state_dict(), '{}/decb_{}.tm'.format(dirname, code))
    torch.save(D_A.state_dict(), '{}/deca_{}.tm'.format(dirname, code))



