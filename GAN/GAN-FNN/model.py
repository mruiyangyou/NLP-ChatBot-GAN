import torch
import torch.nn as nn
import numpy as np

# loss
loss = nn.BCELoss()

#generator
class Generator(nn.Module):
    def __init__(self,args):
        super(Generator, self).__init__()
        self.args = args
        def block(in_channels, out_channels, normalize = True):
            layers = [nn.Linear(in_channels, out_channels)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_channels, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.args['input_dim'],128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            *block(1024, int(np.prod(self.args['img_shape']))),
            nn.Tanh()
        )

    def forward(self,x):
        return self.model(x)


# discrimiantor

class Discriminator(nn.Module):
     def __init__(self, args):
         super(Discriminator, self).__init__()
         self.model = nn.Sequential(
             nn.Linear(np.product(args['img_shape']), 512),
             nn.LeakyReLU(0.2),
             nn.Linear(512, 256),
             nn.LeakyReLU(0.2),
             nn.Linear(256,1),
             nn.Sigmoid()
         )

     def forward(self, x):
         return self.model(x)


args = {'input_dim':256, 'img_shape':np.array([1, 28, 28])}

generator = Generator(args)
discriminator = Discriminator(args)

input_d = torch.randn(8, 28*28)
input_g = torch.randn(8, 256)
print('Start:')
print("Generator:", generator(input_g).shape)
print('Dis:', discriminator(input_d), discriminator(input_d).shape)







