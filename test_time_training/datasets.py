import numpy as np
import torch

from torch.utils.data import TensorDataset
from torchvision.datasets import CIFAR10
from torchvision.transforms.functional import rotate


def rotated_dataset(images, method='all', return_dataset=True, tensor_embedding=False):
    """ make a dataset of rotated images """
    all_images = []
    all_angles = []

    angles = [0, 90, -90, 180]
    for image in images:
        if method == 'all':
            rotations = angles
        else:
            rotations = [np.random.choice(angles)]

        for angle in rotations:
            all_images.append(rotate(image, int(angle)))
            all_angles.append(angle)

    all_images = torch.stack(all_images)
    all_angles = torch.as_tensor(all_angles)

    if tensor_embedding:
        # embed as tensors
        angles_radians = (2*np.pi / 360) * all_angles
        all_angles = torch.stack([torch.cos(angles_radians),
                                 torch.sin(angles_radians)], dim=-1)
    else:
        # embed angles as class labels
        indices = {angle: i for i, angle in enumerate(angles)}
        all_angles = torch.as_tensor(list(map(lambda x: indices[x.item()], all_angles)))

    if not return_dataset:
        return all_images, all_angles

    return TensorDataset(all_images, all_angles)


# Corrupted CIFAR datasets
corruption_names = ['none', 'brightness', 'gaussian_noise', 'saturate', 'contrast',
                    'glass_blur', 'shot_noise', 'defocus_blur', 'impulse_noise',
                    'snow', 'elastic_transform', 'jpeg_compression', 'spatter',
                    'fog', 'labels', 'speckle_noise', 'frost', 'motion_blur',
                    'zoom_blur', 'gaussian_blur', 'pixelate']


def load_corrupted(name, severity=1, root_dir='/Users/luke/Datasets',
                   transform=lambda x: x):
    """ Load data with specific synthetic corruption """
    assert severity <= 5

    cifar_vanilla = CIFAR10(root=root_dir, train=False, transform=transform)
    targets = cifar_vanilla.targets

    if name == 'none':
        full_data = cifar_vanilla.data / 256
        selection = full_data
    else: 
        full_data = np.load(f'{root_dir}/CIFAR-10-C/{name}.npy') / 256
        selection = full_data[10000*(severity - 1):10000*severity]
        
    outputs = transform(torch.as_tensor(selection).permute([0, 3, 1, 2]).to(torch.float32))

    return TensorDataset(outputs,
                         torch.as_tensor(targets)
                         )
