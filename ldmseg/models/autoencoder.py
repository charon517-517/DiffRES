import torch
import torch.nn as nn
class AutoEncoder(nn.Module):
    def __init__(self, n_class=1, dim=128, out_dim=4):
        super().__init__()
        self.n_class = n_class
        self.dim = dim
        self.encoder = nn.Sequential(
            nn.Conv2d(n_class,dim,4,2,1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim,dim,4,2,1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim,out_dim,4,2,1),
            nn.BatchNorm2d(out_dim),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(out_dim, dim,4,2,1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim,dim,4,2,1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim,n_class,4,2,1),
            nn.Sigmoid(),
        )
    def forward(self, img):
        hiddens = self.encoder(img)
        latent = self.decoder(hiddens)
        return latent

    def encode(self, img):
        return self.encoder(img)
    
    def decode(self, latent):
        return self.decoder(latent)
    
if __name__ == "__main__":
    ae = AutoEncoder()
    img = torch.rand(8,1,512,512)
    # latent = ae.encode(img)
    out = ae(img)
    # print(latent.shape)
    print(out.shape)
    