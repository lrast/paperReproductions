# generates TTA curves
import numpy as np

from training import test_time_adaptation
from model import ClassifierWithTTA

from safetensors import safe_open


def TTA_curve(model_layers, dataset, device='cuda:0'):
    """ Full TTA curve for a random subset of the dataset
    """
    folder = {
                0: "linear_probe",
                2: "vit2_probe",
                4: "vit4_probe"
             }.get(model_layers)

    state_dict = {}
    with safe_open(f"{folder}/checkpoint-62500/model.safetensors", framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

    model = ClassifierWithTTA(classifier_hidden_layers=model_layers)

    results = np.zeros(7)

    for i in range(len(dataset)):
        model.load_state_dict(state_dict)

        inputs = dataset.tensors[0][i][None, :, :, :]
        labels = dataset.tensors[1][i]

        sample = test_time_adaptation(model, inputs, labels=labels,
                                      steps=30, evaluate_freq=5,
                                      device=device)
        results += sample

    return results
