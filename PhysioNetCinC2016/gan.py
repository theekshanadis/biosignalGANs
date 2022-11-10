import torch
import torch.nn.functional as F
from torch import nn
import torchaudio
import math
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

plt.style.use('ggplot')

class Dataset(torch.utils.data.Dataset):
    def __init__(self, indexes, test=False):
        self.indexes = indexes

    def __getitem__(self, index):
        x = np.load(self.indexes[index])
        x = torch.from_numpy(x).unsqueeze(dim=0).float()

        transform = torchaudio.transforms.Resample(2000, 500)
        x = transform(x)

        x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
        x = torch.nan_to_num(x)

        return x.squeeze(dim=0)

    def __len__(self):
        return self.indexes.shape[0]

class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = torch.nn.Conv1d(1, 12, kernel_size=3, bias=True)
        self.conv2 = torch.nn.Conv1d(12, 12, kernel_size=3, bias=True)
        self.conv3 = torch.nn.Conv1d(12, 24, kernel_size=3, bias=True)
        self.conv4 = torch.nn.Conv1d(24, 24, kernel_size=3, bias=True)
        self.conv5 = torch.nn.Conv1d(24, 32, kernel_size=3, bias=True)
        self.conv6 = torch.nn.Conv1d(32, 32, kernel_size=3, bias=True)
        self.conv7 = torch.nn.Conv1d(32, 48, kernel_size=3, bias=True)
        self.conv8 = torch.nn.Conv1d(48, 48, kernel_size=3, bias=True)

        self.mx_1 = torch.nn.MaxPool1d(kernel_size=2, return_indices=True)
        self.mx_2 = torch.nn.MaxPool1d(kernel_size=2, return_indices=True)
        self.mx_3 = torch.nn.MaxPool1d(kernel_size=4, return_indices=True)
        self.mx_4 = torch.nn.MaxPool1d(kernel_size=2, return_indices=True)
        self.mx_5 = torch.nn.MaxPool1d(kernel_size=2, return_indices=True)

        self.linear1 = torch.nn.Linear(5 * 48, 200)
        self.linear2 = torch.nn.Linear(200, 1)

    def forward(self, x):
        x = x.unsqueeze(dim=1)

        x = self.conv1(x)
        x = F.relu(x)
        x = F.dropout(x, 0.4)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.dropout(x, 0.4)
        x, id1 = self.mx_1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.dropout(x, 0.4)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.dropout(x, 0.4)
        x, id2 = self.mx_2(x)

        x = self.conv5(x)
        x = F.relu(x)
        x = F.dropout(x, 0.4)
        x = self.conv6(x)
        x = F.relu(x)
        x = F.dropout(x, 0.4)
        x, id3 = self.mx_3(x)

        x = self.conv7(x)
        x = F.relu(x)
        x = F.dropout(x, 0.4)
        x, id4 = self.mx_4(x)

        x = self.conv8(x)
        x = F.relu(x)
        x = F.dropout(x, 0.4)
        x, id5 = self.mx_5(x)

        x = self.linear1(x.flatten(start_dim=1)).relu()
        x = F.dropout(x, 0.3)

        x = self.linear2(x)

        return torch.sigmoid(x)

class Generator(torch.nn.Module):
    def __init__(self, z):
        super(Generator, self).__init__()

        self.z = z
        self.conv1 = torch.nn.ConvTranspose1d(12, 1, kernel_size=3, bias=True)
        self.conv2 = torch.nn.ConvTranspose1d(12, 12, kernel_size=3, bias=True)
        self.conv3 = torch.nn.ConvTranspose1d(24, 12, kernel_size=3, bias=True)
        self.conv4 = torch.nn.ConvTranspose1d(24, 24, kernel_size=3, bias=True)
        self.conv5 = torch.nn.ConvTranspose1d(32, 24, kernel_size=3, bias=True)
        self.conv6 = torch.nn.ConvTranspose1d(32, 32, kernel_size=3, bias=True)
        self.conv7 = torch.nn.ConvTranspose1d(48, 32, kernel_size=3, bias=True)
        self.conv8 = torch.nn.ConvTranspose1d(48, 48, kernel_size=3, bias=True)

        self.mx_1 = torch.nn.MaxUnpool1d(kernel_size=2)
        self.mx_2 = torch.nn.MaxUnpool1d(kernel_size=2)
        self.mx_3 = torch.nn.MaxUnpool1d(kernel_size=4)
        self.mx_4 = torch.nn.MaxUnpool1d(kernel_size=2)
        self.mx_5 = torch.nn.MaxUnpool1d(kernel_size=2)

        self.linear1 = torch.nn.Linear(200, 5 * 48)
        self.linear2 = torch.nn.Linear(z, 200)

    def forward(self, x, id1, id2, id3, id4, id5):
        x = self.linear2(x)
        x = F.dropout(x.relu(), 0.4)

        x = self.linear1(x).relu()
        x = F.dropout(x, 0.4).view(-1, 48, 5)

        x = self.mx_5(x, id5, output_size=(-1, 48, 11))
        x = self.conv8(x).relu()
        x = F.dropout(x, 0.4)

        x = self.mx_4(x, id4, output_size=(-1, 48, 27))
        x = self.conv7(x).relu()
        x = F.dropout(x, 0.4)

        x = self.mx_3(x, id3, output_size=(-1, 32, 118))
        x = self.conv6(x)
        x = F.relu(x)
        x = F.dropout(x, 0.4)
        x = self.conv5(x)
        x = F.relu(x)
        x = F.dropout(x, 0.4)

        x = self.mx_2(x, id2, output_size=(-1, 24, 244))
        x = self.conv4(x)
        x = F.relu(x)
        x = F.dropout(x, 0.4)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.dropout(x, 0.4)

        x = self.mx_1(x, id1, output_size=(-1, 12, 496))
        x = self.conv2(x)
        x = F.relu(x)
        x = F.dropout(x, 0.4)
        x = self.conv1(x)
        
        return x.squeeze(dim=1).sigmoid()

# data loader
import glob
data = np.array(glob.glob('./Data/D3/*'))

X_train, X_test = train_test_split(data, test_size=0.01, random_state=42)
print('train-on', X_train.shape, 'test-on', X_test.shape)

# data loaders
train_dataset = Dataset(X_train)
test_dataset = Dataset(X_test)

params_te = {'batch_size': 100,
             'shuffle': False,
             'num_workers': 8}

training_generator = torch.utils.data.DataLoader(train_dataset, **params_te)
validation_generator = torch.utils.data.DataLoader(test_dataset, **params_te)

# model
z = 64
generator = Generator(z=z)
discriminator = Discriminator()
code = 'GAN-T2V5-{:d}'.format(z)

generator, discriminator = generator.cuda(), discriminator.cuda()

adverserial_loss = nn.BCELoss()

optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

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

for epoch in range(200):
    print('Epoch---{:d}---{}'.format(epoch, code))

    traind_loss, traing_loss = [], []
    for i, x in enumerate(training_generator):    
        # adverserial ground-truths
        valid = Variable(torch.ones((x.shape[0], 1)), requires_grad=False).cuda()
        fake = Variable(torch.zeros((x.shape[0], 1)), requires_grad=False).cuda()

        # ---- Train Generator ------
        optimizer_G.zero_grad()
        
        z = Variable(torch.from_numpy(np.random.normal(0, 1, (x.shape[0], generator.z)))).float()
        id1, id2, id3, id4, id5 = torch.load('ids.tm')

        if id1.shape[0] > x.shape[0]:
            bs = x.shape[0]
            id1, id2, id3, id4, id5 = id1[:bs, ], id2[:bs, ], id3[:bs, ], id4[:bs, ], id5[:bs, ]

        x_recons = generator(z.cuda(), id1.cuda(), id2.cuda(), id3.cuda(), id4.cuda(), id5.cuda())
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
        if i == 1:
            ids = np.random.choice(range(100), 50, replace=True)
            for k in range(50):
                _ = plt.figure(figsize=(6, 3))
                plt.plot(x_recons[ids[k], ].detach().cpu().numpy())
                plt.savefig('./SIN/T2/V5/f_{:d}.png'.format(k))
                plt.close()
                plt.clf()

    print('\n--------Train generator: {:.4f} discriminator: {:.4f}'.format(np.mean(traing_loss), np.mean(traind_loss)))

    generator_loss.append(np.mean(traing_loss))
    discriminator_loss.append(np.mean(traind_loss))

    torch.save(generator.state_dict(), './Models/gen_{}.tm'.format(code))
    torch.save(discriminator.state_dict(), './Models/dis_{}.tm'.format(code))
    torch.save(optimizer_D.state_dict(), './Models/opt_d_{}.tm'.format(code))
    torch.save(optimizer_G.state_dict(), './Models/opt_g_{}.tm'.format(code))

    _ = plt.figure(figsize=(6, 3))
    plt.plot(range(epoch+1), generator_loss)
    plt.plot(range(epoch+1), discriminator_loss)
    plt.legend(['generator', 'discriminator'])
    plt.ylabel('loss')
    plt.savefig('./Loss/loss-{}.png'.format(code))
    plt.close()
    plt.clf()

    
