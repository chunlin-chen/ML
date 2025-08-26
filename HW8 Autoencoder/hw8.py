# pip install -q qqdm

import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.optim import Adam, AdamW
from qqdm import qqdm, format_str
import pandas as pd


train = np.load('trainingset.npy', allow_pickle=True)
test = np.load('testingset.npy', allow_pickle=True)

print(train.shape)
print(test.shape)

def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

same_seeds(48763)

class fcn_autoencoder(nn.Module):
    def __init__(self):
        super(fcn_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64 * 64 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(), 
            nn.Linear(64, 12), 
            nn.ReLU(), 
            nn.Linear(12, 3) 
        ) # dimension of latent space can be adjusted
        
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(), 
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(), 
            nn.Linear(128, 64 * 64 * 3), 
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class conv_autoencoder(nn.Module):
    def __init__(self):
        super(conv_autoencoder, self).__init__()
        self.encoder = nn.Sequential( # size: 64*64
            nn.Conv2d(3, 12, 4, stride=2, padding=1),         
            nn.ReLU(), # size: 32*32
            nn.Conv2d(12, 24, 4, stride=2, padding=1),        
            nn.ReLU(), # size: 16*16
			nn.Conv2d(24, 48, 4, stride=2, padding=1),         
            nn.ReLU(), # size: 8*8
        ) # dimansion of latent space can be adjusted
        self.decoder = nn.Sequential(
			nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),
            nn.ReLU(),
			nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),            
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),    
            nn.ReLU(),
        )
        self.enc_out_1 = nn.Sequential(
            nn.Conv2d(24, 48, 4, stride=2, padding=1),  
            nn.ReLU(),
        )
        self.enc_out_2 = nn.Sequential(
            nn.Conv2d(24, 48, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        # add more layer to encoder and decoder
        self.decoder = nn.Sequential(
			nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1), 
            nn.ReLU(),
			nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1), 
            nn.Tanh(),
        )

    def encode(self, x):
        h1 = self.encoder(x)
        return self.enc_out_1(h1), self.enc_out_2(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiEncoderAutoencoder(nn.Module):
    def __init__(self):
        super(MultiEncoderAutoencoder, self).__init__()
        
        # Fully-Connected Encoder
        self.fcn_encoder = nn.Sequential(
            nn.Linear(64 * 64 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 6),
            nn.ReLU(),
            nn.Linear(6, 3)
            
        )
        
        # Convolutional Encoder
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),
            nn.ReLU(), # 48*8*8
            nn.Conv2d(48, 96, 4, stride=2, padding=1),
            nn.ReLU(), # 96*4*4
            nn.Conv2d(96, 192, 4, stride=2, padding=1),
            nn.ReLU() # 192*2*2
        )
        
        # VAE Encoder
        self.vae_encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 96, 4, stride=2, padding=1),
            nn.ReLU()
        )
        self.vae_mu = nn.Conv2d(96, 192, 4, stride=2, padding=1)
        self.vae_logvar = nn.Conv2d(96, 192, 4, stride=2, padding=1)

        # Decoder (shared)     
        self.decoder = nn.Sequential(
            nn.Linear(192 * 2 * 2 * 2 + 3, 192 * 2 * 2),
            nn.ReLU(),
            nn.Unflatten(1, (192, 2, 2)),  # Reshape to match CNN/VAE latent dimensions
            nn.ConvTranspose2d(192, 96, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        # Flatten input for FCN
        x_flat = x.view(x.size(0), -1)
        
        # FCN Encoder
        fcn_latent = self.fcn_encoder(x_flat)  # Output shape: [batch, 6]
        # print(f"Shape of fcn_latent: {fcn_latent.shape}")
        
        # CNN Encoder
        cnn_latent = self.cnn_encoder(x)  # Output shape: [batch, 48, 8, 8]
        # print(f"Shape of cnn_latent: {cnn_latent.shape}")
        
        # VAE Encoder
        vae_h = self.vae_encoder(x)  # Intermediate latent space
        vae_mu = self.vae_mu(vae_h)  # Mean
        vae_logvar = self.vae_logvar(vae_h)  # Log-variance
        vae_latent = self.reparametrize(vae_mu, vae_logvar)  # Reparametrization trick
        # print(f"Shape of vae_latent: {vae_latent.shape}")

        # Combine latent spaces
        # Flatten CNN and VAE latent spaces to concatenate with FCN
        cnn_latent_flat = cnn_latent.view(cnn_latent.size(0), -1)  # [batch, 48*8*8]
        vae_latent_flat = vae_latent.view(vae_latent.size(0), -1)  # [batch, 48*8*8]
        combined_latent = torch.cat((fcn_latent, cnn_latent_flat, vae_latent_flat), dim=1)
        # print(f"Shape of combined_latent: {combined_latent.shape}")

        '''
        # Optionally, add a fusion layer
        fusion_layer = nn.Linear(combined_latent.size(1), 48 * 8 * 8).to(device)
        fused_latent = fusion_layer(combined_latent)
        # print(f"Shape of fused_latent: {fused_latent.shape}")

        # Reshape fused latent space for decoder
        fused_latent_reshaped = fused_latent.view(-1, 48, 8, 8)
        # print(f"Shape of fused_reshaped_latent: {fused_latent_reshaped.shape}")
        '''
        
        # Decode
        reconstructed = self.decoder(combined_latent)
        # print(f"Shape of reconstructed: {reconstructed.shape}")
        
        return reconstructed, vae_mu, vae_logvar



def loss_vae(recon_x, x, mu, logvar, criterion):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    mse = criterion(recon_x, x)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return mse + KLD


class CustomTensorDataset(TensorDataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors):
        self.tensors = tensors
        if tensors.shape[-1] == 3:
            self.tensors = tensors.permute(0, 3, 1, 2)
        
        self.transform = transforms.Compose([
          transforms.Lambda(lambda x: x.to(torch.float32)),
          transforms.Lambda(lambda x: 2. * x/255. - 1.),
        ])
        
    def __getitem__(self, index):
        x = self.tensors[index]
        
        if self.transform:
            # mapping images to [-1.0, 1.0]
            x = self.transform(x)

        return x

    def __len__(self):
        return len(self.tensors)

# Training hyperparameters
num_epochs = 50
batch_size = 1000 # batch size may be lower
learning_rate = 1e-3

# Build training dataloader
x = torch.from_numpy(train)
train_dataset = CustomTensorDataset(x)

train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

# Model
model_type = 'multi'   # selecting a model type from {'cnn', 'fcn', 'vae', 'resnet'}
model_classes = {'fcn': fcn_autoencoder(), 'cnn': conv_autoencoder(), 'vae': VAE(), 'multi': MultiEncoderAutoencoder()}
model = model_classes[model_type].cuda()

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


best_loss = np.inf
model.train()

qqdm_train = qqdm(range(num_epochs), desc=format_str('bold', 'Description'))
for epoch in qqdm_train:
    tot_loss = list()
    for data in train_dataloader:

        # ===================loading=====================
        img = data.float().cuda()
        if model_type in ['fcn']:
            img = img.view(img.shape[0], -1)

        # ===================forward=====================
        output = model(img)
        if model_type in ['vae']:
            loss = loss_vae(output[0], img, output[1], output[2], criterion)
        elif model_type in ['multi']:
            loss_1 = loss_vae(output[0], img, output[1], output[2], criterion)
            loss_2 = criterion(output[0], img)
            loss = loss_1*0.8 + loss_2*0.2
        else:
            loss = criterion(output, img)

        tot_loss.append(loss.item())
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================save_best====================
    mean_loss = np.mean(tot_loss)
    if mean_loss < best_loss:
        best_loss = mean_loss
        torch.save(model, 'best_model_{}_strong.pt'.format(model_type))
        print('epoch:', f'{epoch + 1:.0f}/{num_epochs:.0f}')
        print('loss:', f'{mean_loss:.4f}')
    # ===================log========================
    qqdm_train.set_infos({
        'epoch': f'{epoch + 1:.0f}/{num_epochs:.0f}',
        'loss': f'{mean_loss:.4f}',
    })
    # ===================save_last========================
    torch.save(model, 'last_model_{}_strong.pt'.format(model_type))


eval_batch_size = 200

# build testing dataloader
data = torch.tensor(test, dtype=torch.float32)
test_dataset = CustomTensorDataset(data)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=eval_batch_size, num_workers=1)
eval_loss = nn.MSELoss(reduction='none')

# load trained model
checkpoint_path = f'last_model_{model_type}_strong.pt'
model = torch.load(checkpoint_path)
model.eval()

# prediction file 
out_file = './prediction_strong.csv'

anomality = list()
with torch.no_grad():
  for i, data in enumerate(test_dataloader):
    img = data.float().cuda()
    if model_type in ['fcn']:
        img = img.view(img.shape[0], -1)

    output = model(img)
    if model_type in ['vae']:
        output = output[0]
    if model_type in ['multi']:
        output = output[0]


    if model_type in ['fcn']:
        loss = eval_loss(output, img).sum(-1)
    else:
        loss = eval_loss(output, img).sum([1, 2, 3])
    anomality.append(loss)
anomality = torch.cat(anomality, axis=0)
anomality = torch.sqrt(anomality).reshape(len(test), 1).cpu().numpy()

df = pd.DataFrame(anomality, columns=['score'])
df.to_csv(out_file, index_label = 'ID')