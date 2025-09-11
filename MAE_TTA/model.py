# models built on huggingface transformers backbones

import torch

from transformers import AutoModelForImageClassification, ViTConfig, ViTMAEForPreTraining, ViTImageProcessor
from transformers.modeling_outputs import ImageClassifierOutput

from torchvision.transforms.v2 import Normalize, Compose, ToDtype


class CustomMAE(ViTMAEForPreTraining):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, pixel_values, *args, **kwargs):
        outputs = super().forward(pixel_values, *args, **kwargs)
        # add auxilliary loss functions here
        return outputs

    def embedding(self, pixel_values, *args, **kwargs):
        self.disable_masking()
        return self.vit(pixel_values, *args, **kwargs).last_hidden_state

    def disable_masking(self):
        self.vit.config.mask_ratio = 0.0

    def enable_masking(self):
        self.vit.config.mask_ratio = 0.75


class ClassifierWithTTA(torch.nn.Module):
    """ 
        Classifier model with embedding for TTA.
            The Embedding is from an MAE model, while the classifier places 
            classifier_hidden_layers number of ViT layers on top of that 
        
        Used as follows:
             1. This model is trained and use`d as a classifier.
             2. TTA is performed by training the embedding model directly.
             3. Both models expect torch tensors pixel encoded as uint8
    """
    def __init__(self, classifier_hidden_layers=12, classifier_kwargs={},
                 embedding_kwargs={}):
        super().__init__()

        self.embedding = CustomMAE.from_pretrained("facebook/vit-mae-base",
                                                   **embedding_kwargs)
        
        _ = classifier_kwargs.setdefault('num_labels', 200)
        class_config = ViTConfig.from_pretrained("google/vit-base-patch16-224",
                                                 num_hidden_layers=classifier_hidden_layers,
                                                 **classifier_kwargs
                                                 )
        self.classifier = AutoModelForImageClassification.from_config(class_config)
        del self.classifier.vit._modules['embeddings']

        processor = ViTImageProcessor.from_pretrained("facebook/vit-mae-base",
                                                      do_convert_rgb=False,
                                                      do_normalize=True,
                                                      do_rescale=True,
                                                      do_resize=False,
                                                      use_fast=True
                                                      )

        self.preprocess = make_online_transform(processor)

    def forward(self, pixel_values, labels=None, **kwargs):
        pixel_values = self.preprocess(pixel_values)
        x = self.embedding.embedding(pixel_values)
        x = self.classifier.vit.encoder(x).last_hidden_state
        x = self.classifier.vit.layernorm(x)[:, 0, :]
        logits = self.classifier.classifier(x)
        
        loss = None
        if labels is not None:
            loss = self.classifier.loss_function(labels, logits, self.classifier.config, **kwargs)

        return ImageClassifierOutput(
                                        loss=loss,
                                        logits=logits
                                     )
    
    def disable_masking(self):
        self.vit.config.mask_ratio = 0.0

    def enable_masking(self):
        self.vit.config.mask_ratio = 0.75

    def freeze_embedding(self):
        for parameter in self.embedding.parameters():
            parameter.requires_grad = False

    def unfreeze_all(self):
        for parameter in self.parameters():
            parameter.requires_grad = True


def make_online_transform(transform):
    equivalent = Compose([
                         ToDtype(torch.float32, scale=True),
                         Normalize(mean=transform.image_mean, std=transform.image_std)
                         ])
    return equivalent
