import torch
import torch.nn.functional as F
from torch import nn
import torchaudio
import math
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import warnings

plt.style.use('ggplot')
warnings.filterwarnings("ignore", category=UserWarning)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, indexes, test=False):
        self.indexes = indexes

    def normalize(self, x):
        return -1 + 2*(x - np.min(x))/(np.max(x) - np.min(x))

    def normalize_(self, x):
        return (x - np.min(x))/(np.max(x) - np.min(x))

    def __getitem__(self, index):
        x = np.load(self.indexes[index]).astype(float)[0, ]
        x = torch.from_numpy(x).unsqueeze(dim=0).float()

        stft_f = torchaudio.transforms.Spectrogram(n_fft=32, hop_length=11)

        x_spec = stft_f(x)
        spec_mag, spec_ang = x_spec.abs(), x_spec.angle()

        return spec_mag

    def __len__(self):
        return self.indexes.shape[0]


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.z = 100
        self.generate = nn.Sequential(
            nn.ConvTranspose2d(100, 256, kernel_size=(2, 4), stride=1, padding=0),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=(2, 4), stride=2, padding=(0, 1)),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=(2, 4), stride=2, padding=(0, 1)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 1, kernel_size=(3, 5), stride=2, padding=(0, 1))
        )

    def forward(self, x):
        return self.generate(x.view(-1, 100, 1, 1))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.discrim = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 5), stride=2, padding=(0, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 128, kernel_size=(2, 4), stride=2, padding=(0, 1)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=(2, 4), stride=2, padding=(0, 1)),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=(2, 4), stride=2, padding=0)
        )

    def forward(self, x):
        return self.discrim(x).flatten(start_dim=1).sigmoid()

# data loader
import glob
data = np.array(glob.glob('./Data/D2/*'))

X_train, X_test = train_test_split(data, test_size=0.01, random_state=42)
print('train-on', X_train.shape, 'test-on', X_test.shape)

# data loaders
train_dataset = Dataset(X_train)
test_dataset = Dataset(X_test)

params_te = {'batch_size': 48,
             'shuffle': False,
             'num_workers': 8}

training_generator = torch.utils.data.DataLoader(train_dataset, **params_te)
validation_generator = torch.utils.data.DataLoader(test_dataset, **params_te)

# create ./Loss/ for save the loss-variation.

# model
z = 256
DIR = 'T3/V1' # create this directory before runing the code. 
generator = Generator()
discriminator = Discriminator()
code = 'GAN-D2-{}-{:d}'.format(DIR.replace('/','-') ,z)

generator, discriminator = generator.cuda(), discriminator.cuda()

adverserial_loss = nn.BCELoss()

optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.001)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001)

try:
    generator.load_state_dict(torch.load('./Models/g_{}.tm'.format(code)))
    optimizer_G.load_state_dict(torch.load('./Models/g_opt_{}.tm'.format(code)))
    
    discriminator.load_state_dict(torch.load('./Models/d_{}.tm'.format(code)))
    optimizer_D.load_state_dict(torch.load('./Models/d_opt_{}.tm'.format(code)))
    
    print('loaded-saved: {}'.format(code))
except Exception as e:    
    print(e)

print('# of parameters (gen):', sum(p.numel() for p in generator.parameters() if p.requires_grad))
print('# of parameters (disc):', sum(p.numel() for p in discriminator.parameters() if p.requires_grad))

min_recons = np.inf
generator_loss, discriminator_loss = [], []

for epoch in range(400):
    print('Epoch---{:d}---{}'.format(epoch, code))

    traind_loss, traing_loss = [], []
    for i, x in enumerate(training_generator):    
        # adverserial ground-truths
        valid = Variable(torch.ones((x.shape[0], 1)), requires_grad=False).cuda()
        fake = Variable(torch.zeros((x.shape[0], 1)), requires_grad=False).cuda()

        # ---- Train Generator ------
        optimizer_G.zero_grad()
        
        z = Variable(torch.from_numpy(np.random.normal(0, 1, (x.shape[0], generator.z)))).float()

        x_recons = generator(z.cuda())
        g_loss = adverserial_loss(discriminator(x_recons), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---- Train Discriminator
        optimizer_D.zero_grad()

        real_loss = adverserial_loss(discriminator(x.cuda()), valid)
        fake_loss = adverserial_loss(discriminator(x_recons.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()
    
        traind_loss.append(d_loss.item())
        traing_loss.append(g_loss.item())
        print('\r[{:4d}] loss: generator: {:.4f} discriminator: {:.4f}'.format(i, g_loss.item(), d_loss.item()), end='')

        # save some images
        if i == 2000:            
            dirname = './SIN/{}/f{:d}'.format(DIR, epoch)
            import os 
            try:
                os.mkdir(dirname)
            except FileExistsError:
                pass
            ids = np.random.choice(48, 24, replace=True)
            for i, id_ in enumerate(ids):
                plt.imshow(x_recons[id_, ][0, ].cpu().detach().numpy())
                plt.savefig('{}/f_{:d}.png'.format(dirname, i))
                
                plt.clf()
                plt.close()

            torch.save(x_recons.cpu(), '{}/rec_.tm'.format(dirname))

    print('\n--------Train generator: {:.4f} discriminator: {:.4f}'.format(np.mean(traing_loss), np.mean(traind_loss)))

    generator_loss.append(np.mean(traing_loss))
    discriminator_loss.append(np.mean(traind_loss))

    torch.save(generator.state_dict(), './Models/generator-{}.tm'.format(code))
    torch.save(discriminator.state_dict(), './Models/discriminator-{}.tm'.format(code))
    torch.save(optimizer_D.state_dict(), './Models/opt_d-{}.tm'.format(code))
    torch.save(optimizer_G.state_dict(), './Models/opt_g-{}.tm'.format(code))

    _ = plt.figure(figsize=(6, 3))
    plt.plot(range(epoch+1), generator_loss)
    plt.plot(range(epoch+1), discriminator_loss)
    plt.legend(['generator', 'discriminator'])
    plt.ylabel('loss')
    plt.savefig('./Loss/loss-{}.png'.format(code))
    plt.close()
    plt.clf()
