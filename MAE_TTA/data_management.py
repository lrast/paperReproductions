# custom data management to get around huggingface's poor handling of image data
import datasets
import torch
import multiprocessing

import numpy as np

from tqdm import tqdm
from transformers import ViTImageProcessor

from torch.utils.data import TensorDataset, random_split


def fetch_dataset_from_hf(dataset_name='zh-plus/tiny-imagenet',
                          processor_name='facebook/vit-mae-base'):
    dataset = datasets.load_dataset(dataset_name)
    processor_initial = ViTImageProcessor.from_pretrained(processor_name,
                                                          do_convert_rgb=True,
                                                          do_normalize=False,
                                                          do_rescale=False,
                                                          do_resize=True
                                                          )
    # avoid using dataset.map for preprocessing: it slows later retrieval.

    train_img_np = [processor_initial(dataset['train'][i]['image'])['pixel_values'][0]
                    for i in range(len(dataset['train']))
                    ]
    train_images = torch.from_numpy(np.stack(train_img_np))

    val_img_np = [processor_initial(dataset['valid'][i]['image'])['pixel_values'][0]
                  for i in range(len(dataset['valid']))
                  ]
    val_images = torch.from_numpy(np.stack(val_img_np))

    train_labels = torch.tensor(dataset['train']['label'])
    val_labels = torch.tensor(dataset['valid']['label'])

    train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    val_dataset = torch.utils.data.TensorDataset(val_images, val_labels)

    return train_dataset, val_dataset


def process_image(image):
    return np.asarray(image.convert('RGB').resize((224, 224)), dtype=np.uint8)


def make_imageset(dataset_name, split):
    """ Load huggingface dataset into memory and return corresponding
        pytorch DataSet

        Saves all images as numpy files.
    """
    datasets.disable_caching()
    raw_data = datasets.load_dataset(dataset_name, split=split)

    pool = multiprocessing.Pool(8)

    images = list(tqdm(
                    pool.imap(process_image, raw_data['image']),
                    total=len(raw_data),
                    desc="Processing Data"
                    ))
    pool.close()

    labels = np.array(list(raw_data['label']))
    images = np.stack(images)

    np.savez(f'pixel_processed_{split}',
             image=images, label=labels)


def fetch_dataset(filename='', split=None, device='cpu', pinned=False):
    """ Returns a Tensor dataset on the device requested. """
    data = np.load(filename)
    images = torch.as_tensor(data['image']).permute([0, 3, 1, 2]).to(device)
    labels = torch.as_tensor(data['label']).to(device)

    if pinned:
        images = images.pin_memory()
        labels = labels.pin_memory()

    dataset = TensorDataset(images, labels)

    if split is None:
        return dataset

    return random_split(dataset, split)


def consistent_subset(dataset, count=1000, seed=42):
    """ Sample a consistent subset of images, balanced between indices. """
    generator = torch.Generator()
    generator.manual_seed(42)

    labels = dataset.tensors[1]
    samples_per = count // len(labels.unique())

    inds = []
    for label in labels.unique():
        locations = torch.where(labels == label)[0]
        loc_i = torch.randperm(len(locations), generator=generator)[0:samples_per]
        inds.extend(locations[loc_i].tolist())

    return torch.utils.data.TensorDataset(dataset.tensors[0][inds].clone(), dataset.tensors[1][inds].clone())
