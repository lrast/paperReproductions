import torch
import pytorch_lightning as pl

from torch import nn
from collections import OrderedDict
from torchvision.models import resnet50, ResNet50_Weights

from datasets import rotated_dataset


class TTTModel(pl.LightningModule):
    """ Version of a backbone model that implements Test Time Training
        on a specific 
    """
    def __init__(self, branch_layer='layer2', train_mode='base_only',
                 target_embedding='angular', n_classes=10
                 ):
        super(TTTModel, self).__init__()
        self.save_hyperparameters()

        self.primary = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.secondary = resnet50(weights=ResNet50_Weights.DEFAULT)

        # setup angle decoding model by mixing the two resnets together
        module_names = list(self.primary._modules.keys())
        branch_ind = module_names.index(branch_layer)

        TTTBranch = OrderedDict([
                            (key, self.primary._modules[key]) if i <= branch_ind
                            else (key, self.secondary._modules[key])
                            for i, key in enumerate(module_names)
                            ])
        # slight quirk of the Resnet implementation: it doesn't work as a
        # simple series of modules. We need to flatten
        TTTBranch['flatten'] = nn.Flatten()
        TTTBranch.move_to_end('fc')
        self.TTTBranch = nn.Sequential(TTTBranch)

        # decoders 
        self.class_decoder = nn.Linear(1000, n_classes)

        angle_dims = (2 if target_embedding == 'angular' else 4)
        self.angle_decoder = nn.Linear(1000, angle_dims)

        # losses
        self.classification_loss = nn.CrossEntropyLoss()

        if target_embedding == 'angular':
            self.angle_loss = lambda x, y: nn.functional.cosine_similarity(x, y).mean()
        else:
            self.angle_loss = nn.CrossEntropyLoss()

        # mode for training
        self.train_mode = train_mode

    def forward(self, x):
        return self.class_decoder(self.primary(x))

    def forward_branch(self, x):
        return self.angle_decoder(self.TTTBranch(x))

    # train, val, test logic
    def training_step(self, batch, batchind=None):
        x, y = batch
        if self.train_mode == 'test_time':
            loss = self.angle_loss(self.forward_branch(x), y)

        if self.train_mode == 'base_only':
            loss = self.classification_loss(self.forward(x), y)

        if self.train_mode == 'joint':
            classification_loss = self.classification_loss(self.forward(x), y)

            rotated, angle = rotated_dataset(x,  method='sample', return_dataset=False,
                                             tensor_embedding=(
                                                self.hparams.target_embedding == 'angular'
                                                )
                                             )
            angle_loss = self.angle_loss(self.forward_branch(rotated), angle.to(self.device))

            loss = classification_loss + angle_loss

        self.log('Train loss', loss)
        return loss

    def validation_step(self, batch, batch_ind=None):
        x, y = batch
        outputs = self.forward(x)
        predictions = torch.argmax(outputs, axis=1)
        accuracy = (predictions == y).to(torch.float32).mean()
        
        loss = self.classification_loss(outputs, y)
        self.log('Val loss', loss)
        self.log('Val accuracy', accuracy)

    def test_step(self, batch, batch_ind=None):
        x, y = batch
        outputs = self.forward(x)
        predictions = torch.argmax(outputs, axis=1)
        accuracy = (predictions == y).to(torch.float32).mean()

        self.log('test accuracy', accuracy)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    # utilities for freezing different parts of the model.
    def freeze_primary(self):
        for parameter in self.primary.parameters():
            parameter.requires_grad = False

    def unfreeze_primary(self):
        for parameter in self.primary.parameters():
            parameter.requires_grad = True

    def freeze_secondary(self):
        for parameter in self.secondary.parameters():
            parameter.requires_grad = False
    
    def unfreeze_secondary(self):
        for parameter in self.secondary.parameters():
            parameter.requires_grad = True


class GroupNormResnet(object):
    """GroupNormResnet: a version of resnet50 that replaces the batch norm layers
       with group norm layers
    """
    def __init__(self, arg):
        super(GroupNormResnet, self).__init__()
        self.arg = arg

