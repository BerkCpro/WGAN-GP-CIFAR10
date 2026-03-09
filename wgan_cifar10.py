
import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as datasets
from tqdm import tqdm
import torchvision.utils as vutils
import wandb

wandb.login()

wandb.init(
    project="CIFAR10-WGAN",
    name="Deneme_5",
    config= {"learning_rate": 1e-4,
        "architecture": "WGAN-GP",
        "dataset": "CIFAR10",
        "epochs": 50,}
)

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
data = datasets.CIFAR10(root="data",transform=transform,download=True)
loader = DataLoader(data,32,shuffle=True)

images, labels = next(iter(loader))
print(images.shape)

class Discriminator(nn.Module):
  def __init__(self,img_dim,features_d):
    super().__init__()
    self.disc = nn.Sequential(
        nn.Conv2d(img_dim,features_d,kernel_size=4,stride=2,padding=1), # 32x32 -> 16x16
        nn.LeakyReLU(0.2),

        self._block(features_d,features_d*2,4,2,1), # 16x16 -> 8x8
        self._block(features_d*2,features_d*4,4,2,1), # 8x8 -> 4x4

        nn.Conv2d(features_d*4,1,kernel_size=4,stride=2,padding=0), # 4x4 -> 1x1

    )

  def _block(self,in_channels,out_channels,kernel_size,stride,padding):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=False),
        nn.InstanceNorm2d(out_channels,affine=True),
        nn.LeakyReLU(0.2))

  def forward(self,x):
    return self.disc(x)

class Generator(nn.Module):
  def __init__(self,z_dim,img_dim,features_g):
    super().__init__()
    self.gen = nn.Sequential(
        self._block(z_dim,features_g*8,4,1,0), # 1x1 -> 2x2
        self._block(features_g*8,features_g*4,4,2,1), # 4x4 -> 8x8
        self._block(features_g*4,features_g*2,4,2,1), # 8x8 -> 16x16

        nn.ConvTranspose2d(features_g*2,img_dim,kernel_size=4,stride=2,padding=1), # 16x16 -> 32x32
        nn.Tanh()
    )

  def _block(self,in_channels,out_channels,kernel_size,stride,padding):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride,padding,bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

  def forward(self,x):
    return self.gen(x)

def initialize_weights(model):
  for m in model.modules():
    if isinstance(m,(nn.Conv2d,nn.ConvTranspose2d,nn.BatchNorm2d)):
      nn.init.normal_(m.weight.data,0.0,0.02)

def gradient_penalty(critic,real,fake,device):
  BATCH_SIZE,C,H,W = real.shape
  epsilon = torch.rand((BATCH_SIZE,1,1,1)).repeat(1,C,H,W).to(device)
  interpolated_images = real*epsilon + fake*(1-epsilon)
  interpolated_images.requires_grad_(True)
  mixed_scores = critic(interpolated_images)

  gradient = torch.autograd.grad(
      inputs = interpolated_images,
      outputs = mixed_scores,
      grad_outputs = torch.ones_like(mixed_scores),
      create_graph=True,
      retain_graph=True
  )[0]

  gradient = gradient.view(gradient.shape[0],-1)
  gradient_norm = gradient.norm(2,dim=1)
  gradient_penalty = torch.mean((gradient_norm-1)**2)
  return gradient_penalty

device = "cuda" if torch.cuda.is_available() else "cpu"

z_dim =64

critic = Discriminator(3,64).to(device)
gen = Generator(z_dim,3,64).to(device)

fixed_noise = torch.randn(32,z_dim,1,1).to(device)

critic.apply(initialize_weights)
gen.apply(initialize_weights)

optimizer_critic = torch.optim.Adam(critic.parameters(),lr=1e-4,betas=(0.0,0.9))
optimizer_gen = torch.optim.Adam(gen.parameters(),lr=1e-4,betas=(0.0,0.9))

epochs = 50
LAMBDA_GP = 10

for epoch in tqdm(range(epochs)):
  for batch_idx,(real,_) in enumerate(loader):
    real= real.to(device)

    for _ in range(5):
      noise = torch.randn(real.shape[0],z_dim,1,1).to(device)
      fake = gen(noise)
      gp = gradient_penalty(critic,real,fake.detach(),device)

      critic_real = critic(real).reshape(-1)
      critic_fake = critic(fake).reshape(-1)

      loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp

      critic.zero_grad()
      loss_critic.backward(retain_graph=True)
      optimizer_critic.step()

    ### Train Generator
    output = critic(fake).reshape(-1)

    loss_gen = -torch.mean(output)

    gen.zero_grad()
    loss_gen.backward()
    optimizer_gen.step()

    if batch_idx % 100 == 0:
      with torch.no_grad():
        gen.eval()
        fake = gen(fixed_noise)

        img_grid = vutils.make_grid(fake,normalize=True)

        wandb.log({
            "Epochs": epoch,
            "Disc_loss": loss_critic.item(),
            "Gen_loss": loss_gen.item(),
            "Images": wandb.Image(img_grid)
        })
        gen.train()

wandb.finish()