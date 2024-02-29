import torch
import torch.nn as nn
import torchvision
import math
import snntorch as snn
from snntorch import utils
from snntorch import surrogate

def get_SpikeSEE_autoencoder(cfg):

    # init encoder and decoder
    encoder = SpikeSEE_Encoder(in_channels=cfg['in_channels'],
                               n_electrodes=cfg['n_electrodes'],
                               out_scaling=cfg['output_scaling'],
                               out_activation=cfg['encoder_out_activation']).to(cfg['device'])

    decoder = SpikeSEE_Decoder(out_channels=cfg['out_channels'],
                               out_activation=cfg['decoder_out_activation']).to(cfg['device'])

    # If output steps are specified, add safety layer at the end of the encoder model
    if cfg['output_steps'] != 'None':
        assert cfg['encoder_out_activation'] == 'sigmoid'
        encoder.output_scaling = 1.0
        encoder = torch.nn.Sequential(encoder,
                                      SafetyLayer(n_steps=10,
                                                  order=2,
                                                  out_scaling=cfg['output_scaling'])).to(cfg['device'])

    return encoder, decoder

def get_MVH_autoencoder(cfg):

    # init encoder and decoder
    encoder = SpikeNN_Encoder(in_channels=cfg['in_channels'],
                               n_electrodes=cfg['n_electrodes'],
                               out_scaling=cfg['output_scaling'],
                               out_activation=cfg['encoder_out_activation']).to(cfg['device'])

    decoder = SpikeNN_Decoder(out_channels=cfg['out_channels'],
                              n_electrodes=cfg['n_electrodes'],
                               out_activation=cfg['decoder_out_activation']).to(cfg['device'])

    # If output steps are specified, add safety layer at the end of the encoder model
    if cfg['output_steps'] != 'None':
        assert cfg['encoder_out_activation'] == 'sigmoid'
        encoder.output_scaling = 1.0
        encoder = torch.nn.Sequential(encoder,
                                      SafetyLayer(n_steps=10,
                                                  order=2,
                                                  out_scaling=cfg['output_scaling'])).to(cfg['device'])

    return encoder, decoder


def get_vanilla_autoencoder(cfg):
    spike_grad = surrogate.atan(alpha=2.0)
    # init encoder and decoder
    encoder = Vanilla_SNN_Encoder(spike_grad=spike_grad,
                               in_channels=cfg['in_channels'],
                               n_electrodes=cfg['n_electrodes'],
                               out_scaling=cfg['output_scaling'],
                               out_activation=cfg['encoder_out_activation']).to(cfg['device'])

    decoder = Vanilla_SNN_Decoder(spike_grad=spike_grad,
                                 in_channels=cfg['in_channels'],
                                 out_channels=cfg['out_channels'],
                               n_electrodes=cfg['n_electrodes'],
                               out_scaling=cfg['output_scaling'],
                               out_activation=cfg['encoder_out_activation']).to(cfg['device'])

    return encoder, decoder

def get_vanilla_autoencoder_extended(cfg):
    spike_grad = surrogate.atan(alpha=2.0)
    # init encoder and decoder
    encoder = Vanilla_SNN_Encoder_STIM(spike_grad=spike_grad,
                               in_channels=cfg['in_channels'],
                               n_electrodes=cfg['n_electrodes'],
                               out_scaling=cfg['output_scaling'],
                               out_activation=cfg['encoder_out_activation']).to(cfg['device'])

    decoder = E2E_Decoder(# spike_grad=spike_grad,
                                 in_channels=cfg['in_channels'],
                                 out_channels=cfg['out_channels'],
                               # n_electrodes=cfg['n_electrodes'],
                               #  out_scaling=cfg['output_scaling'],
                               out_activation=cfg['encoder_out_activation']).to(cfg['device'])

    return encoder, decoder

def get_vanilla_autoencoder_extended_test(cfg):
    spike_grad = surrogate.atan(alpha=2.0)
    # init encoder and decoder
    encoder = Vanilla_SNN_Encoder_STIM_SPLIT2(spike_grad=spike_grad,
                               in_channels=cfg['in_channels'],
                               n_electrodes=cfg['n_electrodes'],
                               out_scaling=cfg['output_scaling'],
                               out_activation=cfg['encoder_out_activation']).to(cfg['device'])

    decoder = E2E_Decoder(# spike_grad=spike_grad,
                                 in_channels=cfg['in_channels'],
                                 out_channels=cfg['out_channels'],
                               # n_electrodes=cfg['n_electrodes'],
                               #  out_scaling=cfg['output_scaling'],
                               out_activation=cfg['encoder_out_activation']).to(cfg['device'])

    return encoder, decoder


def get_e2e_autoencoder(cfg):

    # initialize encoder and decoder
    encoder = E2E_Encoder(in_channels=cfg['in_channels'],
                          n_electrodes=cfg['n_electrodes'],
                          out_scaling=cfg['output_scaling'],
                          out_activation=cfg['encoder_out_activation']).to(cfg['device'])

    decoder = E2E_Decoder(out_channels=cfg['out_channels'],
                          out_activation=cfg['decoder_out_activation']).to(cfg['device'])
    
    # If output steps are specified, add safety layer at the end of the encoder model 
    if cfg['output_steps'] != 'None':
        assert cfg['encoder_out_activation'] == 'sigmoid'
        encoder.output_scaling = 1.0
        encoder = torch.nn.Sequential(encoder,
                                      SafetyLayer(n_steps=10,
                                                  order=2,
                                                  out_scaling=cfg['output_scaling'])).to(cfg['device'])
    return encoder, decoder

def get_Zhao_autoencoder(cfg):
    encoder = ZhaoEncoder(in_channels=cfg['in_channels'], n_electrodes=cfg['n_electrodes']).to(cfg['device'])
    decoder = ZhaoDecoder(out_channels=cfg['out_channels'], out_activation=cfg['decoder_out_activation']).to(cfg['device'])

    return encoder, decoder

def get_beta_autoencoder(cfg):
    encoder = SpikeNN_Encoder_Pehuen(in_channels=cfg['in_channels'], n_electrodes=cfg['n_electrodes']).to(cfg['device'])
    decoder = E2E_Decoder(  # spike_grad=spike_grad,
        in_channels=cfg['in_channels'],
        out_channels=cfg['out_channels'],
        # n_electrodes=cfg['n_electrodes'],
        #  out_scaling=cfg['output_scaling'],
        out_activation=cfg['encoder_out_activation']).to(cfg['device'])

    return encoder, decoder

def convlayer(n_input, n_output, k_size=3, stride=1, padding=1, resample_out=None):
    layer = [
        nn.Conv2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(n_output),
        nn.LeakyReLU(inplace=True),
        resample_out]
    if resample_out is None:
        layer.pop()
    return layer


def convlayer3d(n_input, n_output, k_size=3, stride=1, padding=1, resample_out=None):
    layer = [
        nn.Conv3d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm3d(n_output),
        nn.LeakyReLU(inplace=True),
        resample_out]
    if resample_out is None:
        layer.pop()
    return layer 

def deconvlayer3d(n_input, n_output, k_size=2, stride=2, padding=0, dilation=1, resample_out=None):
    layer = [
        nn.ConvTranspose3d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm3d(n_output),
        nn.LeakyReLU(inplace=True),
        resample_out]
    if resample_out is None:
        layer.pop()
    return layer


class ResidualBlock(nn.Module):
    def __init__(self, n_channels, stride=1, resample_out=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, n_channels,kernel_size=3, stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_channels, n_channels,kernel_size=3, stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(n_channels)
        self.resample_out = resample_out

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        if self.resample_out:
            out = self.resample_out(out)
        return out
    
  
class SafetyLayer(torch.nn.Module):
    def __init__(self, n_steps=5, order=1, out_scaling=120e-6):
        super(SafetyLayer, self).__init__()
        self.n_steps = n_steps
        self.order = order
        self.output_scaling = out_scaling

    def stairs(self, x):
        """Assumes input x in range [0,1]. Returns quantized output over range [0,1] with n quantization levels"""
        return torch.round((self.n_steps-1)*x)/(self.n_steps-1)

    def softstairs(self, x):
        """Assumes input x in range [0,1]. Returns sin(x) + x (soft staircase), scaled to range [0,1].
        param n: number of phases (soft quantization levels)
        param order: number of recursion levels (determining the steepnes of the soft quantization)"""

        return (torch.sin(((self.n_steps - 1) * x - 0.5) * 2 * math.pi) +
                         (self.n_steps - 1) * x * 2 * math.pi) / ((self.n_steps - 1) * 2 * math.pi)
    
    def forward(self, x):
        out = self.softstairs(x) + self.stairs(x).detach() - self.softstairs(x).detach()
        return (out * self.output_scaling).clamp(1e-32,None)


class VGGFeatureExtractor():
    def __init__(self,layer_names=['1','3','6','8'], layer_depth=9 ,device='cpu'):
        
        # Load the VGG16 model
        model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
        self.feature_extractor = torch.nn.Sequential(*[*model.features][:layer_depth]).to(device)
        
        # Register a forward hook for each layer of interest
        self.layers = {name: layer for name, layer in self.feature_extractor.named_children() if name in layer_names}
        self.outputs = dict()
        for name, layer in self.layers.items():
            layer.__name__ = name
            layer.register_forward_hook(self.store_output)
            
    def store_output(self, layer, input, output):
        self.outputs[layer.__name__] = output

    def __call__(self, x):
        
        # If grayscale, convert to RGB
        if x.shape[1] == 1:
            x = x.repeat(1,3,1,1)
        
        # Forward pass
        self.feature_extractor(x)
        activations = list(self.outputs.values())
        
        return activations


class E2E_Encoder(nn.Module):
    """
    Simple non-generic encoder class that receives 128x128 input and outputs 32x32 feature map as stimulation protocol
    """
    def __init__(self, in_channels=3, out_channels=1, n_electrodes=638, out_scaling=1e-4, out_activation='relu'):
        super(E2E_Encoder, self).__init__()
        self.output_scaling = out_scaling
        self.out_activation = {'tanh': nn.Tanh(), ## NOTE: simulator expects only positive stimulation values 
                               'sigmoid': nn.Sigmoid(),
                               'relu': nn.ReLU(),
                               'softmax':nn.Softmax(dim=1)}[out_activation]

        # Model
        self.model = nn.Sequential(*convlayer(in_channels,8,3,1,1),
                                   *convlayer(8,16,3,1,1,resample_out=nn.MaxPool2d(2)),
                                   *convlayer(16,32,3,1,1,resample_out=nn.MaxPool2d(2)),
                                   ResidualBlock(32, resample_out=None),
                                   ResidualBlock(32, resample_out=None),
                                   ResidualBlock(32, resample_out=None),
                                   ResidualBlock(32, resample_out=None),
                                   *convlayer(32,16,3,1,1),
                                   nn.Conv2d(16,1,3,1,1),
                                   nn.Flatten(),
                                   nn.Linear(1024,n_electrodes),
                                   self.out_activation)

    def forward(self, x):
        self.out = self.model(x)
        stimulation = self.out*self.output_scaling #scaling improves numerical stability
        return stimulation

class E2E_Decoder(nn.Module):
    """
    Simple non-generic phosphene decoder.
    in: (256x256) SVP representation
    out: (128x128) Reconstruction
    """
    def __init__(self, in_channels=1, out_channels=1, out_activation='sigmoid'):
        super(E2E_Decoder, self).__init__()

        # Activation of output layer
        self.out_activation = {'tanh': nn.Tanh(),
                               'sigmoid': nn.Sigmoid(),
                               'relu': nn.LeakyReLU(),
                               'softmax':nn.Softmax(dim=1)}[out_activation]

        # Model
        self.model = nn.Sequential(*convlayer(in_channels,16,3,1,1),
                                   *convlayer(16,32,3,1,1),
                                   *convlayer(32,64,3,2,1),
                                   ResidualBlock(64),
                                   ResidualBlock(64),
                                   ResidualBlock(64),
                                   ResidualBlock(64),
                                   *convlayer(64,32,3,1,1),
                                   nn.Conv2d(32,out_channels,3,1,1),
                                   self.out_activation)

    def forward(self, x):
        return self.model(x)

class ZhaoEncoder(nn.Module):
    def __init__(self, in_channels=3,n_electrodes=638, out_channels=1):
        super(ZhaoEncoder, self).__init__()

        self.model = nn.Sequential(
            *convlayer3d(in_channels,32,3,1,1, resample_out=nn.MaxPool3d(2,(1,2,2),padding=(1,0,0),dilation=(2,1,1))),
            *convlayer3d(32,48,3,1,1, resample_out=nn.MaxPool3d(2,(1,2,2),padding=(1,0,0),dilation=(2,1,1))),
            *convlayer3d(48,64,3,1,1),
            *convlayer3d(64,1,3,1,1),

            nn.Flatten(start_dim=3),
            nn.Linear(1024,n_electrodes),
            nn.ReLU()
        )

    def forward(self, x):
        self.out = self.model(x)
        self.out = self.out.squeeze(dim=1)
        self.out = self.out*1e-4
        return self.out

class ZhaoDecoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, out_activation='sigmoid'):
        super(ZhaoDecoder, self).__init__()
        
        # Activation of output layer
        self.out_activation = {'tanh': nn.Tanh(),
                               'sigmoid': nn.Sigmoid(),
                               'relu': nn.LeakyReLU(),
                               'softmax':nn.Softmax(dim=1)}[out_activation]

        self.model = nn.Sequential(
            *convlayer3d(in_channels,16,3,1,1),
            *convlayer3d(16,32,3,1,1),
            *convlayer3d(32,64,3,(1,2,2),1),
            *convlayer3d(64,32,3,1,1),
            nn.Conv3d(32,out_channels,3,1,1),
            self.out_activation
        )

    def forward(self, x):
        self.out = self.model(x)
        return self.out

class SpikeSEE_ResidualBlock(nn.Module):
    def __init__(self, n_channels, stride=1, resample_out=None):
        super(SpikeSEE_ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, n_channels,kernel_size=3, stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_channels, n_channels,kernel_size=3, stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(n_channels)
        self.resample_out = resample_out

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.resample_out:
            out = self.resample_out(out)
        return out

class SpikeSEE_Encoder(nn.Module):
    """
    Simple non-generic encoder class that receives 128x128 input and outputs 32x32 feature map as stimulation protocol
    """
    def __init__(self, in_channels=3, out_channels=1, n_electrodes=638, out_scaling=1e-4, out_activation='relu'):
        super(SpikeSEE_Encoder, self).__init__()
        self.output_scaling = out_scaling
        self.out_activation = {'tanh': nn.Tanh(), ## NOTE: simulator expects only positive stimulation values
                               'sigmoid': nn.Sigmoid(),
                               'relu': nn.ReLU(),
                               'softmax': nn.Softmax(dim=1)}[out_activation]

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=25, stride=1, padding=12, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(16, 4, kernel_size=25, stride=1, padding=12, bias=False)
        self.bn2 = nn.BatchNorm2d(4)
        self.relu2 = nn.LeakyReLU()

        self.conv3 = nn.Conv2d(4, 2, kernel_size=25, stride=1, padding=12, bias=False)
        self.bn3 = nn.BatchNorm2d(2)
        self.relu3 = nn.LeakyReLU()

        self.res1 = SpikeSEE_ResidualBlock(in_channels)
        self.res2 = SpikeSEE_ResidualBlock(16, resample_out=nn.Conv2d(16, 2, kernel_size=1, stride=1, bias=False))

        self.lin1 = nn.Linear(32768, n_electrodes)

    def forward(self, x):
        res1 = self.res1(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        res2 = self.res2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out += res1
        out += res2
        out = torch.flatten(out, start_dim=1)
        out = self.lin1(out)
        self.out = self.out_activation(out)
        stimulation = self.out * self.output_scaling # scaling improves numerical stability

        # print("POST STACKING")
        # print(stimulation.size())

        return stimulation


class SpikeSEE_Decoder(nn.Module):
    """
    Simple non-generic phosphene decoder.
    """
    def __init__(self, in_channels=1, out_channels=1, out_activation='sigmoid'):
        super(SpikeSEE_Decoder, self).__init__()

        # Activation of output layer
        self.out_activation = {'tanh': nn.Tanh(),
                               'sigmoid': nn.Sigmoid(),
                               'relu': nn.LeakyReLU(),
                               'softmax':nn.Softmax(dim=1)}[out_activation]

        # Model
        self.model = nn.Sequential(*convlayer(in_channels,16,3,1,1),
                                   *convlayer(16,32,3,1,1),
                                   *convlayer(32,64,3,2,1),
                                   ResidualBlock(64),
                                   ResidualBlock(64),
                                   ResidualBlock(64),
                                   ResidualBlock(64),
                                   *convlayer(64,32,3,1,1),
                                   nn.Conv2d(32,out_channels,3,1,1),
                                   self.out_activation)

    def forward(self, x):
        return self.model(x)


# Define Network
class SpikeNN_Encoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, n_electrodes=638, out_scaling=1e-4, out_activation='relu', num_steps=25, beta=.95, thresh=1):
        # latent_dim = n_electrodes
        super().__init__()
        self.output_scaling = out_scaling
        self.num_steps = num_steps

        self.out_activation = {'tanh': nn.Tanh(),  ## NOTE: simulator expects only positive stimulation values
                               'sigmoid': nn.Sigmoid(),
                               'relu': nn.ReLU(),
                               'softmax': nn.Softmax(dim=1)}[out_activation]

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=25, stride=1, padding=12, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        # self.relu1 = nn.LeakyReLU()
        self.lif1 = snn.Leaky(beta=beta, threshold=thresh)

        self.conv2 = nn.Conv2d(16, 4, kernel_size=25, stride=1, padding=12, bias=False)
        self.bn2 = nn.BatchNorm2d(4)
        # self.relu2 = nn.LeakyReLU()
        self.lif2 = snn.Leaky(beta=beta, threshold=thresh)

        self.conv3 = nn.Conv2d(4, 2, kernel_size=25, stride=1, padding=12, bias=False)
        self.bn3 = nn.BatchNorm2d(2)
        self.relu3 = nn.LeakyReLU()
        self.lif3 = snn.Leaky(beta=beta, threshold=thresh)

        self.res1 = SpikeSEE_ResidualBlock(in_channels)
        self.res2 = SpikeSEE_ResidualBlock(16, resample_out=nn.Conv2d(16, 2, kernel_size=1, stride=1, bias=False))

        self.lin1 = nn.Linear(32768, n_electrodes)

        self.lif4 = snn.Leaky(beta=beta, output=True, threshold=thresh)

        self.latentToConv = nn.Sequential(nn.Linear(n_electrodes, 128 * 4 * 4),
                                          snn.Leaky(beta=beta, init_hidden=True, output=True, threshold=thresh))

    def encode(self, x, mem):
        out = self.conv1(x)
        out = self.bn1(out)
        out, mem[0] = self.lif1(out, mem[0])
        out = self.conv2(out)
        out = self.bn2(out)
        out, mem[1] = self.lif2(out, mem[1])
        out = self.conv3(out)
        out = self.bn3(out)
        out, mem[2] = self.lif3(out, mem[2])
        # LAST OUT
        out = torch.flatten(out, start_dim=1)
        out = self.lin1(out)
        out, mem[3] = self.lif4(out, mem[3])
        return out, mem[3]
    def forward(self, x):

        # encode
        spk_mem = []
        spk_rec = []

        # init mem
        mem = [self.lif1.init_leaky(),
               self.lif2.init_leaky(),
               self.lif3.init_leaky(),
               self.lif4.init_leaky()
               ]

        for step in range(self.num_steps):  # for t in time
            spk_x,mem_x = self.encode(x, mem)
            spk_rec.append(spk_x)
            spk_mem.append(mem_x)
        spk_rec = torch.stack(spk_rec, dim=0)
        spk_mem = torch.stack(spk_mem, dim=0)
        return spk_rec, spk_mem


class SpikeNN_Decoder(nn.Module):
    """
    Simple non-generic phosphene decoder.
    """
    def __init__(self, in_channels=1, out_channels=1, n_electrodes=638, out_activation='sigmoid', num_steps=25, beta=.95, thresh=1):
        super(SpikeNN_Decoder, self).__init__()
        self.num_steps = num_steps
        # Activation of output layer
        self.out_activation = {'tanh': nn.Tanh(),
                               'sigmoid': nn.Sigmoid(),
                               'relu': nn.LeakyReLU(),
                               'softmax': nn.Softmax(dim=1)}[out_activation]

        self.decoder = nn.Sequential(# nn.Unflatten(1, (128, 4, 4)),  # Unflatten data from 1 dim to tensor of 128 x 4 x 4
                                     snn.Leaky(beta=beta, init_hidden=True, threshold=thresh),
                                     nn.ConvTranspose2d(in_channels, 64, 3, padding=1, stride=(2, 2), output_padding=1),
                                     nn.BatchNorm2d(64),
                                     snn.Leaky(beta=beta, init_hidden=True, threshold=thresh),
                                     nn.ConvTranspose2d(64, 32, 3, padding=1, stride=(2, 2), output_padding=1),
                                     nn.BatchNorm2d(32),
                                     snn.Leaky(beta=beta, init_hidden=True, threshold=thresh),
                                     nn.ConvTranspose2d(32, 1, 3, padding=1, stride=(2, 2), output_padding=1),
                                     snn.Leaky(beta=beta, init_hidden=True, output=True,
                                               threshold=20000)  # make large so membrane can be trained
                                     )

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=25, stride=1, padding=12, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.lif1 = snn.Leaky(beta=beta, threshold=thresh)

        self.conv2 = nn.Conv2d(16, 4, kernel_size=25, stride=1, padding=12, bias=False)
        self.bn2 = nn.BatchNorm2d(4)
        self.lif2 = snn.Leaky(beta=beta, threshold=thresh)

        self.conv3 = nn.Conv2d(4, 2, kernel_size=25, stride=1, padding=12, bias=False)
        self.bn3 = nn.BatchNorm2d(2)
        self.lif3 = snn.Leaky(beta=beta, output=True, threshold=20000)

    def decoder(self, x, mem):
        out = self.conv1(x)
        out = self.bn1(out)
        out, mem[0] = self.lif1(out, mem[0])
        out = self.conv2(out)
        out = self.bn2(out)
        out, mem[1] = self.lif2(out, mem[1])
        out = self.conv3(out)
        out, mem[2] = self.lif3(out, mem[2])
        return out, mem[2]

    def forward(self, spk_rec):

        mem = [self.lif1.init_leaky(),
               self.lif2.init_leaky(),
               self.lif3.init_leaky()
               ]

        spk_mem2 = [];
        spk_rec2 = [];
        for step in range(self.num_steps):
            x_recon, x_mem_recon = self.decoder(spk_rec[step, ...], mem)
            spk_rec2.append(x_recon)
            spk_mem2.append(x_mem_recon)
        spk_rec2 = torch.stack(spk_rec2, dim=4)
        spk_mem2 = torch.stack(spk_mem2, dim=4)
        out = spk_mem2[:, :, :, :, -1]
        return out

class SpikeNN_Encoder_Pehuen(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, n_electrodes=638, out_scaling=1e-4, out_activation='relu',
                 num_steps=25, beta=.95, thresh=1, spike_grad=None):
        super().__init__()
        self.num_timesteps = num_steps

        # Define the spiking layers as per the image architecture
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Corresponds to layer 3 in the architecture

        self.conv2 = nn.Conv2d(in_channels, 32, kernel_size=7, stride=1, padding=0)
        self.lif1 = snn.Leaky(beta=beta, threshold=thresh)

        self.pool2 = nn.MaxPool2d(kernel_size=1, stride=1)  # Corresponds to layer 5 in the architecture

        self.conv3 = nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=0)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=thresh)

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # Corresponds to layer 5 in the architecture

        self.conv5 = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=0)
        self.lif5 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=thresh)

        self.pool4 = nn.MaxPool2d(kernel_size=1, stride=1)  # Corresponds to layer 7 in the architecture

        self.conv7 = nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=0)
        self.lif7 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=thresh)

        self.pool5 = nn.MaxPool2d(kernel_size=1, stride=1)  # Corresponds to layer 7 in the architecture


        # Flatten layer to prepare for the linear layer
        self.spikes_to_stim = nn.Sequential(nn.Flatten(start_dim=1),
                                            nn.Linear(627200, n_electrodes), # 4900, 25088, 627200
                                            nn.ReLU())


    def encode(self, x, mem):
        out = [None,
               None,
               None,
               None,
               None
               ]

        temp_out = self.pool1(x)  # Max pooling does not have state
        out[0], mem[0] = self.lif1(self.conv2(temp_out), mem[0])
        temp_out = self.pool2(out[0])  # Max pooling does not have state
        out[1], mem[1] = self.lif2(self.conv3(temp_out), mem[1])
        temp_out = self.pool3(out[1])
        out[2], mem[2] = self.lif5(self.conv5(temp_out), mem[2])
        temp_out = self.pool4(out[2])
        out[3], mem[3] = self.lif7(self.conv7(temp_out), mem[3])
        out[4] = self.pool5(out[3]) # TAKE CARE: THIS IS THE OUTPUT OF POOLING; NOT LIF

        return out, mem

    def forward(self, x):
        # Initialize the list of membrane potentials for each LIF layer
        mem = [self.lif1.init_leaky(), self.lif2.init_leaky(), self.lif5.init_leaky(),
               self.lif7.init_leaky()]

        # Lists to collect outputs and spikes across all timesteps
        all_out = []
        all_mem = []
        last_out = []

        for timestep in range(self.num_timesteps):
            spikes, mem = self.encode(x, mem)
            all_out.append(spikes)
            all_mem.extend(spikes)
            last_out.append(spikes[-1])

        # Stack outputs and spikes from all timesteps
        end_spikes = torch.stack(last_out, dim=1)
        out = self.spikes_to_stim(end_spikes)

        return out

class Spike_Decoder_STIM(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_electrodes=638, out_scaling=1e-4, out_activation='relu', num_steps=25, beta=.95, thresh=1, spike_grad=None):
        # latent_dim = n_electrodes
        super().__init__()
        self.output_scaling = out_scaling
        self.num_steps = num_steps

        self.out_activation = {'tanh': nn.Tanh(),  ## NOTE: simulator expects only positive stimulation values
                               'sigmoid': nn.Sigmoid(),
                               'relu': nn.ReLU(),
                               'softmax': nn.Softmax(dim=1)}[out_activation]

        # From latent back to tensor for convolution
        self.latentToConv = nn.Sequential(nn.Linear(n_electrodes, 128 * 16 * 16),
                                          snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True,
                                                    threshold=thresh))  # Decoder

        self.decoder = nn.Sequential(
            # First, downscale from 256x256 to 128x128
            nn.Conv2d(in_channels, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=thresh),

            # Now use ConvTranspose2d layers to process further
            nn.ConvTranspose2d(128, 64, 3, padding=1, stride=1),  # Size remains 128x128
            nn.BatchNorm2d(64),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=thresh),

            nn.ConvTranspose2d(64, 32, 3, padding=1, stride=1),  # Size remains 128x128
            nn.BatchNorm2d(32),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=thresh),

            nn.ConvTranspose2d(32, out_channels, 3, padding=1, stride=1),  # Size remains 128x128
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True, threshold=20000)
        )

    def forward(self, spk_rec):
        utils.reset(self.decoder)
        utils.reset(self.latentToConv)

        # decode
        spk_mem2 = []
        spk_rec2 = []
        decoded_x = []
        for step in range(self.num_steps):  # for t in time
            x_recon, x_mem_recon = self.decode(spk_rec[..., step])
            spk_rec2.append(x_recon)
            spk_mem2.append(x_mem_recon)
        spk_rec2 = torch.stack(spk_rec2, dim=4)
        spk_mem2 = torch.stack(spk_mem2, dim=4)
        out = spk_mem2[:, :, :, :, -1]  # return the membrane potential of the output neuron at t = -1 (last t)
        return out

    def decode(self, x):
        # spk_x, mem_x = self.latentToConv(x)  # convert latent dimension back to total size of features in encoder final layer
        spk_x2, mem_x2 = self.decoder(x)
        return spk_x2, mem_x2


class Vanilla_SNN_Encoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, n_electrodes=638, out_scaling=1e-4, out_activation='relu', num_steps=25, beta=.95, thresh=1, spike_grad=None):
        # latent_dim = n_electrodes
        super().__init__()
        self.output_scaling = out_scaling
        self.num_steps = num_steps

        self.out_activation = {'tanh': nn.Tanh(),  ## NOTE: simulator expects only positive stimulation values
                               'sigmoid': nn.Sigmoid(),
                               'relu': nn.ReLU(),
                               'softmax': nn.Softmax(dim=1)}[out_activation]

        self.encoder = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1, stride=2),
                                     nn.BatchNorm2d(32),
                                     snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=thresh),
                                     nn.Conv2d(32, 64, 3, padding=1, stride=2),
                                     nn.BatchNorm2d(64),
                                     snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=thresh),
                                     nn.Conv2d(64, 128, 3, padding=1, stride=2),
                                     nn.BatchNorm2d(128),
                                     snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=thresh),
                                     nn.Flatten(start_dim=1, end_dim=3),
                                     nn.Linear(32768, n_electrodes),
                                     # this needs to be the final layer output size (channels * pixels * pixels)
                                     snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True,
                                               threshold=thresh)
                                     )

        self.output_mapping = nn.Sequential(nn.Flatten(start_dim=1),
                                            nn.Linear(n_electrodes * self.num_steps, n_electrodes),
                                            nn.ReLU())

    def forward(self, x):
        utils.reset(self.encoder)  # need to reset the hidden states of LIF

        # encode
        spk_mem = []
        spk_rec = []
        for step in range(self.num_steps):  # for t in time
            spk_x, mem_x = self.encode(x)  # Output spike trains and neuron membrane states
            spk_rec.append(spk_x)
            spk_mem.append(mem_x)
        spk_rec = torch.stack(spk_rec, dim=2)
        spk_mem = torch.stack(spk_mem, dim=2)

        # out = self.output_mapping(spk_rec)

        return spk_rec

    def encode(self, x):
        spk_latent_x, mem_latent_x = self.encoder(x)
        return spk_latent_x, mem_latent_x

class Vanilla_SNN_Encoder_STIM(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, n_electrodes=638, out_scaling=1e-4, out_activation='relu', num_steps=25, beta=.95, thresh=1, spike_grad=None):
        super().__init__()
        self.output_scaling = out_scaling
        self.num_steps = num_steps

        # THIS ONE IS GOOD - it also worked with an initial out channel of 32 and 16 (each decreasing half each layer)
        self.encoder = nn.Sequential(nn.Conv2d(in_channels, 128, 25, padding=12, stride=2),
                                     nn.BatchNorm2d(128),
                                     snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,
                                               threshold=thresh),
                                     nn.Conv2d(128, 64, 25, padding=12, stride=2),
                                     nn.BatchNorm2d(64),
                                     snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,
                                               threshold=thresh),
                                     nn.Conv2d(64, 32, 25, padding=12, stride=2),
                                     nn.BatchNorm2d(32),
                                     snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,
                                               threshold=thresh),
                                     nn.Conv2d(32, 16, 25, padding=12, stride=2),
                                     nn.BatchNorm2d(16),
                                     snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,
                                               threshold=thresh),
                                     nn.Flatten(start_dim=1),
                                     nn.Linear(1024, n_electrodes),
                                     nn.Softmax()
                                     )

        self.output_mapping = nn.Sequential(nn.Flatten(start_dim=1),
                                            nn.Linear(n_electrodes * self.num_steps, n_electrodes),
                                            nn.ReLU())

    def forward(self, x):
        utils.reset(self.encoder)  # need to reset the hidden states of LIF

        # encode
        stim = None
        for step in range(self.num_steps):
            stim = self.encoder(x)

        return stim

class Vanilla_SNN_Encoder_STIM_SPLIT(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, n_electrodes=638, out_scaling=1e-4, out_activation='relu', num_steps=25, beta=.95, thresh=1, spike_grad=None):
        super().__init__()
        self.output_scaling = out_scaling
        self.num_steps = num_steps

        # THIS ONE IS GOOD - it also worked with an initial out channel of 32 and 16 (each decreasing half each layer)
        self.encoder = nn.Sequential(nn.Conv2d(in_channels, 128, 25, padding=12, stride=2),
                                     nn.BatchNorm2d(128),
                                     snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,
                                               threshold=thresh),
                                     nn.Conv2d(128, 64, 25, padding=12, stride=2),
                                     nn.BatchNorm2d(64),
                                     snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,
                                               threshold=thresh),
                                     nn.Conv2d(64, 32, 25, padding=12, stride=2),
                                     nn.BatchNorm2d(32),
                                     snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,
                                               threshold=thresh),
                                     nn.Conv2d(32, 16, 25, padding=12, stride=2),
                                     nn.BatchNorm2d(16),
                                     snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,
                                               threshold=thresh)
                                     )
                                    # Look at spikies here
                                    # Shapre [1024, 25]
                                    # Visualize as rasterplot

        self.mapper = nn.Sequential(nn.Flatten(start_dim=1),
                                    nn.Linear(1024, n_electrodes),
                                    nn.Softmax()
                                    )


    def forward(self, x):
        utils.reset(self.encoder)  # need to reset the hidden states of LIF
        utils.reset(self.mapper)

        # encode
        stim = None
        for step in range(self.num_steps):
            stim = self.encoder(x)
            stim = self.mapper(stim)

        return stim

    def encode_only(self, x):
        return self.encoder(x)

class Vanilla_SNN_Encoder_STIM_SPLIT2(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, n_electrodes=638, out_scaling=1e-4, out_activation='relu', num_steps=25, beta=.95, thresh=1, spike_grad=None):
        super().__init__()
        self.output_scaling = out_scaling
        self.num_steps = num_steps

        # THIS ONE IS GOOD - it also worked with an initial out channel of 32 and 16 (each decreasing half each layer)
        self.encoder = nn.Sequential(nn.Conv2d(in_channels, 256, 25, padding=12, stride=2),
                                     nn.BatchNorm2d(256),
                                     snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,
                                               threshold=thresh),
                                     nn.Conv2d(256, 64, 25, padding=12, stride=2),
                                     nn.BatchNorm2d(64),
                                     snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,
                                               threshold=thresh),
                                     nn.Conv2d(64, 16, 25, padding=12, stride=2),
                                     nn.BatchNorm2d(16),
                                     snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,
                                               threshold=thresh)
                                     )
                                    # Look at spikies here
                                    # Shapre [1024, 25]
                                    # Visualize as rasterplot

        self.mapper = nn.Sequential(nn.Flatten(start_dim=1),
                                    nn.Linear(4096, n_electrodes),
                                    nn.Softmax()
                                    )


    def forward(self, x):
        utils.reset(self.encoder)  # need to reset the hidden states of LIF
        utils.reset(self.mapper)

        # encode
        stim = None
        for step in range(self.num_steps):
            stim = self.encoder(x)
            stim = self.mapper(stim)

        return stim

    def encode_only(self, x):
        return self.encoder(x)

class Vanilla_SNN_Encoder_STIM_LIN_SPLIT(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, n_electrodes=638, out_scaling=1e-4, out_activation='relu', num_steps=25, beta=.95, thresh=1, spike_grad=None):
        super().__init__()
        self.output_scaling = out_scaling
        self.num_steps = num_steps

        self.mapper = nn.Sequential(nn.Flatten(start_dim=1),
                                     nn.Linear(1024, n_electrodes),
                                     nn.Softmax()
                                     )

    def forward(self, x):
        utils.reset(self.mapper)  # need to reset the hidden states of LIF
        return self.mapper(x)


class Vanilla_SNN_Decoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, n_electrodes=638, out_scaling=1e-4, out_activation='relu', num_steps=25, beta=.95, thresh=1, spike_grad=None):
        super().__init__()
        self.output_scaling = out_scaling
        self.num_steps = num_steps

        self.out_activation = {'tanh': nn.Tanh(),
                               'sigmoid': nn.Sigmoid(),
                               'relu': nn.ReLU(),
                               'softmax': nn.Softmax(dim=1)}[out_activation]

        # From latent back to tensor for convolution
        self.latentToConv = nn.Sequential(nn.Linear(n_electrodes, 128 * 16 * 16),
                                          snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True,
                                                    threshold=thresh))

        self.decoder = nn.Sequential(nn.Unflatten(1, (128, 16, 16)),
                                     snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=thresh),
                                     nn.ConvTranspose2d(128, 64, 3, padding=1, stride=(2, 2), output_padding=1),
                                     nn.BatchNorm2d(64),
                                     snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=thresh),
                                     nn.ConvTranspose2d(64, 32, 3, padding=1, stride=(2, 2), output_padding=1),
                                     nn.BatchNorm2d(32),
                                     snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=thresh),
                                     nn.ConvTranspose2d(32, out_channels, 3, padding=1, stride=(2, 2), output_padding=1),
                                     snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True,
                                               threshold=20000)
                                     )

    def forward(self, spk_rec):
        utils.reset(self.decoder)
        utils.reset(self.latentToConv)

        # decode
        spk_mem2 = []
        spk_rec2 = []
        for step in range(self.num_steps):
            x_recon, x_mem_recon = self.decode(spk_rec[..., step])
            spk_rec2.append(x_recon)
            spk_mem2.append(x_mem_recon)
        spk_rec2 = torch.stack(spk_rec2, dim=4)
        spk_mem2 = torch.stack(spk_mem2, dim=4)
        out = spk_mem2[:, :, :, :, -1]
        return out

    def decode(self, x):
        spk_x, mem_x = self.latentToConv(x)
        spk_x2, mem_x2 = self.decoder(spk_x)
        return spk_x2, mem_x2



