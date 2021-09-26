import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Encoder(nn.Module):
    '''
    def __init__(self):
        super(Encoder,self).__init__()
        self.conv1=nn.Conv2d(3,128,32, padding=2,stride=2)   #in_channels=3
        self.bn1=nn.BatchNorm2d(128,momentum=0.9)
        self.conv2=nn.Conv2d(128,128,5,padding=2,stride=2)
        self.bn2=nn.BatchNorm2d(128,momentum=0.9)
        self.conv3=nn.Conv2d(128,256,5,padding=2,stride=2)
        self.bn3=nn.BatchNorm2d(256,momentum=0.9)
        self.relu=nn.LeakyReLU(0.2)
        self.fc1=nn.Linear(256, 3,32)
        self.bn4=nn.BatchNorm1d(3,momentum=0.9)
        self.fc_mean=nn.Linear(3,32)
        self.fc_logvar=nn.Linear(3,32)   #latent dim=128
  
    def forward(self,x):
        batch_size=x.size()[0]
        out=self.relu(self.bn1(self.conv1(x)))
        out=self.relu(self.bn2(self.conv2(out)))

        print("self.bn3(self.conv3(out))", self.bn3(self.conv3(out)).shape)
        print("qqqqq", self.relu(self.bn3(self.conv3(out))).shape)

        out=self.relu(self.bn3(self.conv3(out)))
        out=out.view(batch_size,-1)
        #print("out", out.shape)
        #print("self.fc1", self.fc1(out).shape)
        #print("self.bn4(self.fc1(out)", self.bn4(self.fc1(out).shape))
        #print("self.bn4(self.fc1(out)", self.bn4(self.fc1(out).shape))
        #print()

        out=self.relu(self.bn4(self.fc1(out)))
        #print("oiut", out.shape)
        mean=self.fc_mean(out)
        logvar=self.fc_logvar(out)
        #print("mean", mean.shape)
        assert 1==123

        return mean,logvar
    '''

    def __init__(self):
        super(Encoder, self).__init__()

        self.n_channel = 3
        self.dim_h = 128
        self.n_z = 1


        self.main = nn.Sequential(
          nn.Conv2d(self.n_channel, self.dim_h, 4, 2, 1, bias=False),
          nn.ReLU(True),
          nn.Conv2d(self.dim_h, self.dim_h * 2, 4, 2, 1, bias=False),
          nn.BatchNorm2d(self.dim_h * 2),
          nn.ReLU(True),
          nn.Conv2d(self.dim_h * 2, self.dim_h * 4, 4, 2, 1, bias=False),
          nn.BatchNorm2d(self.dim_h * 4),
          nn.ReLU(True),
          nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 2, 0, bias=False),
          nn.BatchNorm2d(self.dim_h * 8),
          nn.ReLU(True),
        )
        #self.fc = nn.Linear(self.dim_h * 8, self.n_z)
        self.fc_mean = nn.Linear(self.dim_h * 8, self.n_z)
        self.fc_logvar = nn.Linear(self.dim_h * 8, self.n_z)

    def forward(self, x):
        x = self.main(x)
        x = x.squeeze()
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar


class Decoder(nn.Module):
    '''
    def __init__(self):
        super(Decoder,self).__init__()
        self.fc1=nn.Linear(32,8*8*256)
        self.bn1=nn.BatchNorm1d(8*8*256,momentum=0.9)
        self.relu=nn.LeakyReLU(0.2)
        self.deconv1=nn.ConvTranspose2d(256,256,6, stride=1, padding=2)
        self.bn2=nn.BatchNorm2d(256,momentum=0.9)
        self.deconv2=nn.ConvTranspose2d(256,128,6, stride=1, padding=0)
        self.bn3=nn.BatchNorm2d(128,momentum=0.9)
        self.deconv3=nn.ConvTranspose2d(128,128,3, stride=1, padding=0)
        self.bn4=nn.BatchNorm2d(128,momentum=0.9)
        self.deconv4=nn.ConvTranspose2d(128,3,17, stride=1, padding=0)
        self.tanh=nn.Tanh()

    def forward(self,x):
        batch_size=x.size()[0]
        #print("x11", x.shape)
        x=self.relu(self.bn1(self.fc1(x)))
        #print("x12", x.shape)
        x=x.view(-1,256,8,8)
        #print("x3", x.shape)
        x=self.relu(self.bn2(self.deconv1(x)))
        #print("x4", x.shape)
        #print("self.deconv2(x))",self.deconv2(x).shape)
        #print("self.bn3(self.deconv2(x))",self.bn3(self.deconv2(x)).shape)
        x=self.relu(self.bn3(self.deconv2(x)))
        #print("x5", x.shape)
        x=self.relu(self.bn4(self.deconv3(x)))
        #print("x6", x.shape)
        x=self.tanh(self.deconv4(x))
        #print("x7", x.shape)
        return x
    '''

    def __init__(self):
        super(Decoder, self).__init__()

        self.n_channel = 3
        self.dim_h = 128
        self.n_z = 1

        self.proj = nn.Sequential(
          nn.Linear(self.n_z, self.dim_h * 8 * 7 * 7),
          nn.ReLU()
        )

        self.main = nn.Sequential(
          nn.ConvTranspose2d(self.dim_h * 8, self.dim_h * 4, 4),
          nn.BatchNorm2d(self.dim_h * 4),
          nn.ReLU(True),
          nn.ConvTranspose2d(self.dim_h * 4, self.dim_h * 2, 4),
          nn.BatchNorm2d(self.dim_h * 2),
          nn.ReLU(True),
          nn.ConvTranspose2d(self.dim_h * 2, 3, 8, stride=2),
          nn.Sigmoid()
        )


    def forward(self, x):
        x = self.proj(x)
        x = x.view(-1, self.dim_h * 8, 7, 7)
        x = self.main(x)
        return x


class Discriminator(nn.Module):
    '''
    def __init__(self):
        super(Discriminator,self).__init__()

        self.conv1=nn.Linear(32,32)#nn.Conv2d(3,128, 3, padding=2,stride=1)
        self.relu=nn.LeakyReLU(0.2)
        self.conv2=nn.Linear(32, 32)
        self.bn1=nn.LeakyReLU(0.2)
        self.conv3=nn.Linear(32,32)
        self.bn2=nn.LeakyReLU(0.2)
        self.conv4=nn.Linear(32,32)
        self.bn3=nn.LeakyReLU(0.2)
        self.fc1=nn.Linear(32,512)
        self.bn4=nn.LeakyReLU(0.2)
        self.fc2=nn.Linear(512,1)
        self.sigmoid=nn.Sigmoid()


    def forward(self,x):
        batch_size=x.size()[0]
        #print("x1",x.shape)
        x=self.relu(self.conv1(x))
        #print("x2", x.shape)
        x=self.relu(self.bn1(self.conv2(x)))
        #print("x3", x.shape)
        x=self.relu(self.bn2(self.conv3(x)))
        #print("x4", x.shape)
        x=self.relu(self.bn3(self.conv4(x)))
        #print("x5", x.shape)
        x=x.view(-1,32)
        #print("x6", x.shape)
        x1=x
        x=self.relu(self.bn4(self.fc1(x)))
        #print("x7", x.shape)
        x=self.sigmoid(self.fc2(x))
        #print("x8", x.shape)

        return x

        '''

    def __init__(self):
        super(Discriminator, self).__init__()

        self.n_channel = 3
        self.dim_h = 128
        self.n_z = 1

        self.main = nn.Sequential(
          nn.Linear(self.n_z, self.dim_h * 4),
          nn.ReLU(True),
          nn.Linear(self.dim_h * 4, self.dim_h * 4),
          nn.ReLU(True),
          nn.Linear(self.dim_h * 4, self.dim_h * 4),
          nn.ReLU(True),
          nn.Linear(self.dim_h * 4, self.dim_h * 4),
          nn.ReLU(True),
          nn.Linear(self.dim_h * 4, 1),
          nn.Sigmoid()
        )

    def forward(self, x):
        x = self.main(x)
        return x


class VAE(nn.Module):
  def __init__(self):
    super(VAE,self).__init__()
    self.encoder=Encoder()
    self.decoder=Decoder()
    #self.discriminator=Discriminator()
    #self.encoder.apply(weights_init)
    #self.decoder.apply(weights_init)
    #self.discriminator.apply(weights_init)


  def forward(self,x):
    #bs=x.size()[0]
    #print("x",x.shape)
    #z_mean=self.encoder(x)
    #std = z_logvar.mul(0.5).exp_()
        
    #sampling epsilon from normal distribution
    #from torch.autograd import Variable
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ### x_tilda=self.decoder(z)
    #print("x_tilda",x_tilda.shape)
    #print("z_mean",z_mean.shape)
    #print("z_logvar",z_logvar.shape)
    bs = x.size()[0]
    #vs = x.size()[1]
    z_mean, z_logvar = self.encoder(x)
    std = z_logvar.mul(0.5).exp_()
    #sampling epsilon from normal distribution
    from torch.autograd import Variable
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epsilon = Variable(torch.randn(bs, 1)).to(device)
    z = z_mean + std * epsilon

    x_recon = self.decoder(z)
    #print("x_recon ",x_recon.shape )
      
    return x_recon, z_mean,z_logvar
    
