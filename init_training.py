import os
import torch
import numpy as np
import pickle

import dynaphos
from dynaphos.cortex_models import get_visual_field_coordinates_probabilistically
from dynaphos.simulator import GaussianSimulator as PhospheneSimulator
from dynaphos.utils import get_data_kwargs

import model

import local_datasets
from torch.utils.data import DataLoader
from utils import resize, normalize, undo_standardize, dilation3x3, CustomSummaryTracker

from torch.utils.tensorboard import SummaryWriter


class LossTerm():
    """Loss term that can be used for the compound loss"""

    def __init__(self, name=None, func=torch.nn.functional.mse_loss, arg_names=None, weight=1.):
        self.name = name
        self.func = func  # the loss function
        self.arg_names = arg_names  # the names of the inputs to the loss function
        self.weight = weight  # the relative weight of the loss term


class CompoundLoss():
    """Helper class for combining multiple loss terms. Initialize with list of
    LossTerm instances. Returns dict with loss terms and total loss"""

    def __init__(self, loss_terms):
        self.loss_terms = loss_terms

    def __call__(self, loss_targets):
        """Calculate all loss terms and the weighted sum"""
        self.out = dict()
        self.out['total'] = 0
        for lt in self.loss_terms:
            func_args = [loss_targets[name] for name in lt.arg_names]  # Find the loss targets by their name
            self.out[lt.name] = lt.func(*func_args)  # calculate result and add to output dict
            self.out['total'] += self.out[lt.name] * lt.weight  # add the weighted loss term to the total
        return self.out

    def items(self):
        """return dict with loss tensors as dict with Python scalars"""
        return {k: v.item() for k, v in self.out.items()}


class RunningLoss():
    """Helper class to track the running loss over multiple batches."""

    def __init__(self):
        self.dict = dict()
        self.reset()

    def reset(self):
        self._counter = 0
        for key in self.dict.keys():
            self.dict[key] = 0.

    def update(self, new_entries):
        """Add the current loss values to the running loss"""
        self._counter += 1
        for key, value in new_entries.items():
            if key in self.dict:
                self.dict[key] += value
            else:
                self.dict[key] = value

    def get(self):
        """Get the average loss values (total loss dived by the processed batch count)"""
        out = {key: (value / self._counter) for key, value in self.dict.items()}
        return out


class L1FeatureLoss(object):
    def __init__(self):
        self.feature_extractor = model.VGGFeatureExtractor(device=device)
        self.loss_fn = torch.nn.functional.l1_loss

    def __call__(self, y_pred, y_true, ):
        true_features = self.feature_extractor(y_true)
        pred_features = self.feature_extractor(y_pred)
        err = [self.loss_fn(pred, true) for pred, true in zip(pred_features, true_features)]
        return torch.mean(torch.stack(err))



def get_dataset(cfg):
    if cfg['dataset'] == 'ADE50K':
        trainset, valset = local_datasets.get_ade50k_dataset(cfg)
    elif cfg['dataset'] == 'BouncingMNIST':
        trainset, valset = local_datasets.get_bouncing_mnist_dataset(cfg)
    elif cfg['dataset'] == 'Characters':
        trainset, valset = local_datasets.get_character_dataset(cfg)
    elif cfg['dataset'] == 'MNIST':
        trainset, valset = local_datasets.get_mnist_dataset(cfg)
    elif cfg['dataset'] == 'spiking_MNIST':
        trainset, valset = local_datasets.get_spiking_mnist_dataset(cfg)
        
    trainloader = DataLoader(trainset, batch_size=cfg['batch_size'],shuffle=True, drop_last=True)
    valloader = DataLoader(valset,batch_size=cfg['batch_size'],shuffle=False, drop_last=True)

    example_batch = next(iter(valloader))
    if cfg['dataset'] != 'MNIST' and cfg['dataset'] != 'spiking_MNIST':
        cfg['circular_mask'] = trainset._mask.to(cfg['device'])

    dataset = {'trainset': trainset,
               'valset': valset,
               'trainloader': trainloader,
               'valloader': valloader,
               'example_batch': example_batch}

    return dataset


def get_models(cfg):
    if cfg['model_architecture'] == 'end-to-end-autoencoder':
        encoder, decoder = model.get_e2e_autoencoder(cfg)
        optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=cfg['learning_rate'])
    elif cfg['model_architecture'] == 'zhao-autoencoder':
        encoder, decoder = model.get_Zhao_autoencoder(cfg)
        optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=cfg['learning_rate'])
    elif cfg['model_architecture'] == 'beta-autoencoder':
        encoder, decoder = model.get_beta_autoencoder(cfg)
        optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=cfg['learning_rate'])
    elif cfg['model_architecture'] == 'SpikeSEE-autoencoder':
        encoder, decoder = model.get_SpikeSEE_autoencoder(cfg)
        optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=cfg['learning_rate'])
    elif cfg['model_architecture'] == 'SpikingMVH-autoencoder':
        encoder, decoder = model.get_MVH_autoencoder(cfg)
        optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=cfg['learning_rate'])
    elif cfg['model_architecture'] == 'SpikingMVH-vanilla':
        encoder, decoder = model.get_vanilla_autoencoder(cfg)
        optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=cfg['learning_rate'])
    elif cfg['model_architecture'] == 'SpikingMVH-extended':
        encoder, decoder = model.get_vanilla_autoencoder_extended(cfg)
        optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=cfg['learning_rate'])
    elif cfg['model_architecture'] == 'SpikingMVH-extended-test':
        encoder, decoder = model.get_vanilla_autoencoder_extended_test(cfg)
        optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=cfg['learning_rate'])
    else:
        raise NotImplementedError

    simulator = get_simulator(cfg)

    models = {'encoder' : encoder,
              'decoder' : decoder,
              'optimizer': optimizer,
              'simulator': simulator,}

    return models


def get_simulator(cfg):
    # initialise simulator
    params = dynaphos.utils.load_params(cfg['base_config'])
    params['run'].update(cfg)
    params['thresholding'].update(cfg)
    device = get_data_kwargs(params)['device']

    with open(cfg['phosphene_map'], 'rb') as handle:
        coordinates_visual_field = pickle.load(handle, )
    simulator = PhospheneSimulator(params, coordinates_visual_field)
    cfg['SPVsize'] = simulator.phosphene_maps.shape[-2:]
    return simulator


def get_logging(cfg):
    out = dict()
    out['training_loss'] = RunningLoss()
    out['validation_loss'] = RunningLoss()
    out['tensorboard_writer'] = SummaryWriter(os.path.join(cfg['save_path'], 'tensorboard/'))
    out['training_summary'] = CustomSummaryTracker()
    out['validation_summary'] = CustomSummaryTracker()
    out['example_output'] = CustomSummaryTracker()
    return out

####### ADJUST OR ADD TRAINING PIPELINE BELOW

def get_training_pipeline(cfg):
    if cfg['pipeline'] == 'unconstrained-image-autoencoder':
        forward, lossfunc = get_pipeline_unconstrained_image_autoencoder(cfg)
    elif cfg['pipeline'] == 'MVH-image-autoencoder':
        forward, lossfunc = get_pipeline_SSEE_image_autoencoder(cfg)
    elif cfg['pipeline'] == 'MVH-spiking':
        forward, lossfunc = get_pipeline_spiking_MVH_image(cfg)
    elif cfg['pipeline'] == 'spiking-vanilla':
        forward, lossfunc = get_pipeline_vanilla_spiking(cfg)
    elif cfg['pipeline'] == 'spiking-extended':
        forward, lossfunc = get_pipeline_vanilla_spiking_extended(cfg)
    elif cfg['pipeline'] == 'spiking-extended-test':
        forward, lossfunc = get_pipeline_vanilla_spiking_extended(cfg)
    elif cfg['pipeline'] == 'constrained-image-autoencoder':
        forward, lossfunc = get_pipeline_constrained_image_autoencoder(cfg)
    elif cfg['pipeline'] == 'supervised-boundary-reconstruction':
        forward, lossfunc = get_pipeline_supervised_boundary_reconstruction(cfg)
    elif cfg['pipeline'] == 'unconstrained-video-reconstruction':
        forward, lossfunc = get_pipeline_unconstrained_video_reconstruction(cfg)
    elif cfg['pipeline'] == 'image-autoencoder-interaction-model':
        print('Interaction model not implemented yet, add interaction model manually..')
        forward, lossfunc = get_pipeline_interaction_model(cfg)
    else:
        print(cfg['pipeline'] + 'not supported yet')
        raise NotImplementedError

    return {'forward': forward, 'compound_loss_func': lossfunc}

def get_pipeline_unconstrained_image_autoencoder(cfg):
    def forward(batch, models, cfg, to_cpu=False):
        """Forward pass of the model."""

        # unpack
        encoder = models['encoder']
        decoder = models['decoder']
        simulator = models['simulator']

        # Data manipulation
        image, _ = batch
        unstandardized_image = undo_standardize(image) # image values scaled back to range 0-1

        # Forward pass
        simulator.reset()
        stimulation = encoder(image)
        phosphenes = simulator(stimulation).unsqueeze(1)
        reconstruction = decoder(phosphenes)

        # Output dictionary
        out = {'input':  unstandardized_image, # * cfg['circular_mask'],
               'stimulation': stimulation,
               'phosphenes': phosphenes,
               'reconstruction': reconstruction, # * cfg['circular_mask'],
               'input_resized': resize(unstandardized_image, # * cfg['circular_mask'],
                                       cfg['SPVsize'])}

        if to_cpu:
            # Return a cpu-copy of the model output
            out = {k: v.detach().cpu().clone() for k, v in out.items()}
        return out

    recon_loss = LossTerm(name='reconstruction_loss',
                          func=torch.nn.MSELoss(),
                          arg_names=('reconstruction', 'input'),
                          weight=1 - cfg['regularization_weight'])

    regul_loss = LossTerm(name='regularization_loss',
                          func=torch.nn.MSELoss(),
                          arg_names=('phosphenes', 'input_resized'),
                          weight=cfg['regularization_weight'])

    loss_func = CompoundLoss([recon_loss, regul_loss])

    return forward, loss_func

# written by MVH
def get_pipeline_SSEE_image_autoencoder(cfg):
    def forward(batch, models, cfg, to_cpu=False):
        """Forward pass of the model."""

        # unpack
        encoder = models['encoder']
        decoder = models['decoder']
        simulator = models['simulator']

        # Data manipulation
        image, _ = batch
        unstandardized_image = undo_standardize(image) # image values scaled back to range 0-1

        # Forward pass
        simulator.reset()
        stimulation = encoder(image)
        phosphenes = simulator(stimulation).unsqueeze(1)
        reconstruction = decoder(phosphenes)

        # Output dictionary
        out = {'input':  unstandardized_image,
               'stimulation': stimulation,
               'phosphenes': phosphenes,
               'reconstruction': reconstruction,
               'input_resized': resize(unstandardized_image, cfg['SPVsize'])}

        if to_cpu:
            # Return a cpu-copy of the model output
            out = {k: v.detach().cpu().clone() for k, v in out.items()}
        return out

    recon_loss = LossTerm(name='reconstruction_loss',
                          func=torch.nn.MSELoss(),
                          arg_names=('reconstruction', 'input'),
                          weight=1 - cfg['regularization_weight'])

    regul_loss = LossTerm(name='regularization_loss',
                          func=torch.nn.MSELoss(),
                          arg_names=('phosphenes', 'input_resized'),
                          weight=cfg['regularization_weight'])

    loss_func = CompoundLoss([recon_loss, regul_loss])

    return forward, loss_func

def get_pipeline_spiking_MVH_image(cfg):
    def forward(batch, models, cfg, to_cpu=False):
        """Forward pass of the model."""

        # unpack
        encoder = models['encoder']
        decoder = models['decoder']
        simulator = models['simulator']

        num_steps = cfg['num_steps']

        # Data manipulation
        image, _ = batch
        unstandardized_image = undo_standardize(image) # image values scaled back to range 0-1

        # Forward pass
        simulator.reset()
        stimulation, _ = encoder(image)
        # print("STIMULATION pre-mean -----------------------------")
        # print(stimulation.size())
        # stimulation = torch.mean(stimulation, 0)
        # print("STIMULATION -----------------------------")
        # print(stimulation.size())
        # phosphenes = simulator(stimulation).unsqueeze(1)

        phosphenes = []
        for step in range(num_steps):
            # print("Stimulation size  -----------------------------")
            # print(stimulation[step].size())
            phosphene = simulator(stimulation[step]).unsqueeze(1)  # we need to apply the simulation to every image
            # print("Phosphene size  -----------------------------")
            # print(phosphene.size())

            phosphenes.append(phosphene)
        phosphenes = torch.stack(phosphenes, dim=0)

        reconstruction = decoder(phosphenes)

        # Output dictionary
        out = {'input': unstandardized_image,
               'stimulation': stimulation,
               'phosphenes': phosphenes,
               'reconstruction': reconstruction,
               'input_resized': resize(unstandardized_image, cfg['SPVsize'])}

        if to_cpu:
            # Return a cpu-copy of the model output
            out = {k: v.detach().cpu().clone() for k, v in out.items()}
        return out

    recon_loss = LossTerm(name='reconstruction_loss',
                          func=torch.nn.MSELoss(),
                          arg_names=('reconstruction', 'input'),
                          weight=1 - cfg['regularization_weight'])

    regul_loss = LossTerm(name='regularization_loss',
                          func=torch.nn.MSELoss(),
                          arg_names=('phosphenes', 'input_resized'),
                          weight=cfg['regularization_weight'])

    loss_func = CompoundLoss([recon_loss, regul_loss])

    return forward, loss_func

def get_pipeline_vanilla_spiking(cfg):
    def forward(batch, models, cfg, to_cpu=False):
        """Forward pass of the model."""

        # unpack
        encoder = models['encoder']
        decoder = models['decoder']

        # Data manipulation
        image, _ = batch
        unstandardized_image = undo_standardize(image) # image values scaled back to range 0-1

        stimulation = encoder(image)
        reconstruction = decoder(stimulation)


        # Output dictionary
        out = {'input': unstandardized_image,
               'stimulation': stimulation,
               'phosphenes': None,
               'reconstruction': reconstruction,
               'input_resized': resize(unstandardized_image, cfg['SPVsize'])}

        if to_cpu:
            # Return a cpu-copy of the model output
            out = {k: v.detach().cpu().clone() for k, v in out.items() if v is not None}
        return out

    recon_loss = LossTerm(name='reconstruction_loss',
                          func=torch.nn.MSELoss(),
                          arg_names=('reconstruction', 'input'),
                          weight=1 - cfg['regularization_weight'])

    loss_func = CompoundLoss([recon_loss])

    return forward, loss_func


def get_pipeline_vanilla_spiking_extended(cfg):
    def forward(batch, models, cfg, to_cpu=False):
        """Forward pass of the model."""

        # unpack
        encoder = models['encoder']
        decoder = models['decoder']
        simulator = models['simulator']

        num_steps = cfg['num_steps']

        # Data manipulation
        image, _ = batch
        unstandardized_image = undo_standardize(image) # image values scaled back to range 0-1

        simulator.reset()

        stimulation = encoder(image)
        """
        print("SPIKE SIZE")
        print(spk.size())
        print("MEMBRANE_POT SIZE")
        print(membrane_pot.size())
        """

        # stimulation = membrane_pot[:, :, -1]
        # stimulation[stimulation < 0] = 0

        print("STIMULATION SIZE")
        print(stimulation.size())
        print("STIMULATION")
        print(stimulation)
        """

        print(torch.nansum(stimulation, -1))
        """


        # stimulation_freq = torch.mean(spk, 2)
        # stimulation_ampl = torch.nansum(spk, 2)

        # print("STIMULATION FREQ and AMPL SIZE")
        # print(stimulation_freq.size())
        # print(stimulation_ampl.size())

        # phosphenes = simulator(stimulation_ampl, frequency=stimulation_freq).unsqueeze(1)


        # phosphenes = simulator(stimulation, pulse_width=stimulation_ampl, frequency=stimulation_freq).unsqueeze(1)
        phosphenes = simulator(stimulation).unsqueeze(1)

        """
        print("PHOSPHENES SIZE")
        print(phosphenes.size())
        print("PHOSPHENES")
        print(phosphenes)
        """

        """
        # decode
        phosphenes = []

        for step in range(num_steps):  # for t in time
            phosphene = simulator(stimulation[..., step]).unsqueeze(1)
            phosphenes.append(phosphene)
        phosphenes = torch.stack(phosphenes, dim=4)
        """

        # print("PHOSPHENES SIZE")
        # print(phosphenes.size())

        reconstruction = decoder(phosphenes)

        # print("RECONSTRUCTION SIZE")
        # print(reconstruction.size())

        # Output dictionary
        out = {'input': unstandardized_image,
               'stimulation': stimulation,
               'phosphenes': phosphenes,
               'reconstruction': reconstruction,
               'input_resized': resize(unstandardized_image, cfg['SPVsize'])}

        if to_cpu:
            # Return a cpu-copy of the model output
            out = {k: v.detach().cpu().clone() for k, v in out.items() if v is not None}
        return out

    recon_loss = LossTerm(name='reconstruction_loss',
                          func=torch.nn.MSELoss(),
                          arg_names=('reconstruction', 'input'),
                          weight=1 - cfg['regularization_weight'])

    regul_loss = LossTerm(name='regularization_loss',
                          func=torch.nn.MSELoss(),
                          arg_names=('phosphenes', 'input_resized'),
                          weight=cfg['regularization_weight'])

    loss_func = CompoundLoss([recon_loss, regul_loss])

    return forward, loss_func

def get_pipeline_vanilla_spiking_extended_test(cfg):
    def forward(batch, models, cfg, to_cpu=False):
        """Forward pass of the model."""

        # unpack
        encoder = models['encoder']
        decoder = models['decoder']
        simulator = models['simulator']

        # Data manipulation
        image, _ = batch
        unstandardized_image = undo_standardize(image) # image values scaled back to range 0-1

        simulator.reset()

        stimulation = encoder(image)

        print(stimulation.shape)

        phosphenes = simulator(stimulation).unsqueeze(1)

        reconstruction = decoder(phosphenes)

        # Output dictionary
        out = {'input': unstandardized_image,
               'stimulation': stimulation,
               'phosphenes': phosphenes,
               'reconstruction': reconstruction,
               'input_resized': resize(unstandardized_image, cfg['SPVsize'])}

        if to_cpu:
            # Return a cpu-copy of the model output
            out = {k: v.detach().cpu().clone() for k, v in out.items() if v is not None}
        return out

    recon_loss = LossTerm(name='reconstruction_loss',
                          func=torch.nn.MSELoss(),
                          arg_names=('reconstruction', 'input'),
                          weight=1 - cfg['regularization_weight'])

    regul_loss = LossTerm(name='regularization_loss',
                          func=torch.nn.MSELoss(),
                          arg_names=('phosphenes', 'input_resized'),
                          weight=cfg['regularization_weight'])

    loss_func = CompoundLoss([recon_loss, regul_loss])

    return forward, loss_func

def get_pipeline_constrained_image_autoencoder(cfg):
    def forward(batch, models, cfg, to_cpu=False):
        """Forward pass of the model."""

        # unpack
        encoder = models['encoder']
        decoder = models['decoder']
        simulator = models['simulator']

        # Data manipulation
        image, _ = batch
        unstandardized_image = undo_standardize(image) # image values scaled back to range 0-1

        # Forward pass
        simulator.reset()
        stimulation = encoder(image)
        phosphenes = simulator(stimulation).unsqueeze(1)
        reconstruction = decoder(phosphenes)

        # Output dictionary
        out = {'input':  unstandardized_image * cfg['circular_mask'],
               'stimulation': stimulation,
               'phosphenes': phosphenes,
               'reconstruction': reconstruction * cfg['circular_mask'],
               'input_resized': resize(unstandardized_image * cfg['circular_mask'], cfg['SPVsize'])}
        
        # Sample phosphenes and target at the centers of the phosphenes
        out.update({'phosphene_centers': simulator.sample_centers(phosphenes),
                    'input_centers': simulator.sample_centers(out['input_resized']) })

        if to_cpu:
            # Return a cpu-copy of the model output
            out = {k: v.detach().cpu().clone() for k, v in out.items()}
        return out

    recon_loss = LossTerm(name='reconstruction_loss',
                          func=torch.nn.MSELoss(),
                          arg_names=('reconstruction', 'input'),
                          weight=1 - cfg['regularization_weight'])

    regul_loss = LossTerm(name='regularization_loss',
                          func=torch.nn.MSELoss(),
                          arg_names=('phosphene_centers', 'input_centers'),
                          weight=cfg['regularization_weight'])

    loss_func = CompoundLoss([recon_loss, regul_loss])

    return forward, loss_func


def get_pipeline_supervised_boundary_reconstruction(cfg):
    def forward(batch, models, cfg, to_cpu=False):
        """Forward pass of the model."""

        # unpack
        encoder = models['encoder']
        decoder = models['decoder']
        simulator = models['simulator']

        # Data manipulation
        image, label = batch
        label = dilation3x3(label)

        # Forward pass
        simulator.reset()
        stimulation = encoder(image)
        phosphenes = simulator(stimulation).unsqueeze(1)
        reconstruction = decoder(phosphenes) * cfg['circular_mask']

        # Output dictionary
        out = {'input': image,
               'stimulation': stimulation,
               'phosphenes': phosphenes,
               'reconstruction': reconstruction * cfg['circular_mask'],
               'target': label * cfg['circular_mask'],
               'target_resized': resize(label * cfg['circular_mask'], cfg['SPVsize'],),}

        # Sample phosphenes and target at the centers of the phosphenes
        out.update({'phosphene_centers': simulator.sample_centers(phosphenes) ,
                    'target_centers': simulator.sample_centers(out['target_resized']) })

        if to_cpu:
            # Return a cpu-copy of the model output
            out = {k: v.detach().cpu().clone() for k, v in out.items()}
        return out

    recon_loss = LossTerm(name='reconstruction_loss',
                          func=torch.nn.MSELoss(),
                          arg_names=('reconstruction', 'target'),
                          weight=1 - cfg['regularization_weight'])

    regul_loss = LossTerm(name='regularization_loss',
                          func=torch.nn.MSELoss(),
                          arg_names=('phosphene_centers', 'target_centers'),
                          weight=cfg['regularization_weight'])

    loss_func = CompoundLoss([recon_loss, regul_loss])

    return forward, loss_func


def get_pipeline_unconstrained_video_reconstruction(cfg):
    def forward(batch, models, cfg, to_cpu=False):
        # Unpack
        frames = batch
        encoder = models['encoder']
        decoder = models['decoder']
        simulator = models['simulator']

        # Forward
        simulator.reset()
        stimulation_sequence = encoder(frames).permute(1, 0, 2)  # permute: (Batch,Time,Num_phos) -> (Time,Batch,Num_phos)
        phosphenes = []
        for stim in stimulation_sequence:
            phosphenes.append(simulator(stim))  # simulator expects (Batch, Num_phosphenes)
        phosphenes = torch.stack(phosphenes, dim=1).unsqueeze(dim=1)  # Shape: (Batch, Channels=1, Time, Height, Width)
        reconstruction = decoder(phosphenes)

        out =  {'stimulation': stimulation_sequence,
                'phosphenes': phosphenes,
                'reconstruction': reconstruction * cfg['circular_mask'],
                'input': frames * cfg['circular_mask'],
                'input_resized': resize(frames * cfg['circular_mask'],
                                         (cfg['sequence_length'],*cfg['SPVsize']),interpolation='trilinear'),}

        if to_cpu:
            # Return a cpu-copy of the model output
            out = {k: v.detach().cpu().clone() for k, v in out.items()}

        return out

    recon_loss = LossTerm(name='reconstruction_loss',
                          func=torch.nn.MSELoss(),
                          arg_names=('reconstruction', 'input'),
                          weight=1-cfg['regularization_weight'])

    regul_loss = LossTerm(name='regularization_loss',
                          func=torch.nn.MSELoss(),
                          arg_names=('phosphenes', 'input_resized'),
                          weight=cfg['regularization_weight'])

    loss_func = CompoundLoss([recon_loss, regul_loss])

    return forward, loss_func


def get_pipeline_interaction_model(cfg):
    def forward(batch, models, cfg, to_cpu=False):
        """Forward pass of the model."""

        # unpack
        encoder = models['encoder']
        interaction_model = models['interaction']
        decoder = models['decoder']
        simulator = models['simulator']

        # Data manipulation
        image, _ = batch

        # Forward pass
        simulator.reset()
        stimulation = encoder(image)
        interaction = interaction_model(stimulation)
        phosphenes = simulator(interaction).unsqueeze(1)
        reconstruction = decoder(phosphenes)

        # Output dictionary
        out = {'input':  image * cfg['circular_mask'],
               'stimulation': stimulation,
               'interaction': interaction,
               'phosphenes': phosphenes,
               'reconstruction': reconstruction * cfg['circular_mask'],
               'input_resized': resize(image * cfg['circular_mask'], cfg['SPVsize'])}
        
        # Sample phosphenes and target at the centers of the phosphenes
#         out.update({'phosphene_centers': simulator.sample_centers(phosphenes),
#                     'input_centers': simulator.sample_centers(out['input_resized']) })
        out.update({'phosphene_brightness': simulator.get_state()['brightness'].squeeze(),
                    'input_centers': simulator.sample_centers(out['input_resized']).squeeze()})

        if to_cpu:
            # Return a cpu-copy of the model output
            out = {k: v.detach().cpu().clone() for k, v in out.items()}
        return out

    recon_loss = LossTerm(name='reconstruction_loss',
                          func=torch.nn.MSELoss(),
                          arg_names=('reconstruction', 'input'),
                          weight=1 - cfg['regularization_weight'])

    regul_loss = LossTerm(name='regularization_loss',
                          func=torch.nn.MSELoss(),
                          arg_names=('phosphene_brightness', 'input_centers'),
                          weight=cfg['regularization_weight'])

    loss_func = CompoundLoss([recon_loss, regul_loss])

    return forward, loss_func