import dataclasses
import numpy as np

from foundations import hparams
import models.base
from pruning import base
from pruning.mask import Mask
from pruning.pruned_model import PrunedModel


@dataclasses.dataclass
class PruningHparams(hparams.PruningHparams):
    pruning_fraction: float = 0.2
    pruning_layers_to_ignore: str = None
    pruning_gradient_weight: float = 1.0

    _name = 'Hyperparameters for SNIP'
    _description = 'Hyperparameters that modify the way pruning occurs.'
    _pruning_fraction = 'The fraction of additional weights to prune from the network.'
    _layers_to_ignore = 'A comma-separated list of addititonal tensors that should not be pruned.'
    _gradient_weight = 'Applies elementwise exponentiation on each gradient.'


class Strategy(base.Strategy):
    @staticmethod
    def get_pruning_hparams() -> type:
        return PruningHparams

    @staticmethod
    def prune(pruning_hparams: PruningHparams, trained_model: models.base.Model, current_mask: Mask = None):
        current_mask = Mask.ones_like(trained_model).numpy() if current_mask is None else current_mask.numpy()

        # Determine the number of weights that need to be pruned.
        number_of_remaining_weights = np.sum([np.sum(v) for v in current_mask.values()])
        number_of_weights_to_prune = np.ceil(
            pruning_hparams.pruning_fraction * number_of_remaining_weights).astype(int)

        # Determine which layers can be pruned.
        prunable_tensors = set(trained_model.prunable_layer_names)
        if pruning_hparams.pruning_layers_to_ignore:
            prunable_tensors -= set(pruning_hparams.pruning_layers_to_ignore.split(','))

        # Get the model weights.
        weights = {k: v.clone().cpu().detach().numpy()
                   for k, v in trained_model.state_dict().items()
                   if k in prunable_tensors}

        # print(weights.keys())
        # print('grads', trained_model.grads)
        # print(dir(trained_model))

        # if isinstance(trained_model, PrunedModel):
        #     print(trained_model.model.grads.keys())

        # print(trained_model.grads.keys())

        grads_w_goodnames = dict([(k[6:], v) if k[:6] == 'model.' else (k, v)
                                   for k, v in trained_model.grads.items()])

        grads = {k: v
                 for k, v in grads_w_goodnames.items()
                 if k in prunable_tensors}

        assert sorted(weights.keys()) == sorted(grads.keys())
        # print(grads['fc_layers.0.weight'][0, 0])
        assert pruning_hparams.pruning_gradient_weight >= 0 or pruning_hparams.pruning_gradient_weight == -1, 'Gradient weight must be either geq than 0 or equal to -1.'
        if pruning_hparams.pruning_gradient_weight == -1:
            snip_sensitivities = {k: grads[k] for k in weights.keys()}
        else:
            snip_sensitivities = {k: weights[k] * grads[k]**pruning_hparams.pruning_gradient_weight for k in weights.keys()}

        sensitivity_vector = np.concatenate([v[current_mask[k] == 1] for k, v in snip_sensitivities.items()])
        threshold = np.sort(np.abs(sensitivity_vector))[number_of_weights_to_prune]

        new_mask = Mask({k: np.where(np.abs(v) > threshold, current_mask[k], np.zeros_like(v))
                         for k, v in snip_sensitivities.items()})
        for k in current_mask:
            if k not in new_mask:
                new_mask[k] = current_mask[k]

        return new_mask
