import numpy as np
import torch
import tqdm
from typing import Callable, List
from cam.grad_cam.base_cam import BaseCAM
from cam.grad_cam.utils.find_layers import replace_layer_recursive
from cam.grad_cam.ablation_layer import AblationLayer


class AblationCAM(BaseCAM):
    def __init__(self,
                 model: torch.nn.Module,
                 target_layers: List[torch.nn.Module],
                 use_cuda: bool = False,
                 reshape_transform: Callable = None,
                 ablation_layer: torch.nn.Module = AblationLayer(),
                 batch_size: int = 32,
                 ratio_channels_to_ablate: float = 1.0) -> None:

        super(AblationCAM, self).__init__(model,
                                          target_layers,
                                          use_cuda,
                                          reshape_transform,
                                          uses_gradients=False)
        self.batch_size = batch_size
        self.ablation_layer = ablation_layer
        self.ratio_channels_to_ablate = ratio_channels_to_ablate

    def save_activation(self, module, input, output) -> None:
        self.activations = output

    def assemble_ablation_scores(self,
                                 new_scores: list,
                                 original_score: float ,
                                 ablated_channels: np.ndarray,
                                 number_of_channels: int) -> np.ndarray:
        index = 0
        result = []
        sorted_indices = np.argsort(ablated_channels)
        ablated_channels = ablated_channels[sorted_indices]
        new_scores = np.float32(new_scores)[sorted_indices]

        for i in range(number_of_channels):
            if index < len(ablated_channels) and ablated_channels[index] == i:
                weight = new_scores[index]
                index = index + 1
            else:
                weight = original_score
            result.append(weight)

        return result

    def get_cam_weights(self,
                        input_tensor: torch.Tensor,
                        target_layer: torch.nn.Module,
                        targets: List[Callable],
                        activations: torch.Tensor,
                        grads: torch.Tensor) -> np.ndarray:

        handle = target_layer.register_forward_hook(self.save_activation)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            handle.remove()
            original_scores = np.float32([target(output).cpu().item() for target, output in zip(targets, outputs)])

        ablation_layer = self.ablation_layer
        replace_layer_recursive(self.model, target_layer, ablation_layer)

        number_of_channels = activations.shape[1]
        weights = []
        with torch.no_grad():
            for batch_index, (target, tensor) in enumerate(zip(targets, input_tensor)):
                new_scores = []
                batch_tensor = tensor.repeat(self.batch_size, 1, 1, 1)

                channels_to_ablate = ablation_layer.activations_to_be_ablated(activations[batch_index, :],
                                                                              self.ratio_channels_to_ablate)
                number_channels_to_ablate = len(channels_to_ablate)

                for i in tqdm.tqdm(range(0, number_channels_to_ablate, self.batch_size)):
                    if i + self.batch_size > number_channels_to_ablate:
                        batch_tensor = batch_tensor[:(number_channels_to_ablate - i)]

                    ablation_layer.set_next_batch(input_batch_index=batch_index,
                                                  activations=self.activations,
                                                  num_channels_to_ablate=batch_tensor.size(0))
                    score = [target(o).cpu().item() for o in self.model(batch_tensor)]
                    new_scores.extend(score)
                    ablation_layer.indices = ablation_layer.indices[batch_tensor.size(0):]

                new_scores = self.assemble_ablation_scores(new_scores,
                                                           original_scores[batch_index],
                                                           channels_to_ablate,
                                                           number_of_channels)
                weights.extend(new_scores)

        weights = np.float32(weights)
        weights = weights.reshape(activations.shape[:2])
        original_scores = original_scores[:, None]
        weights = (original_scores - weights) / original_scores

        replace_layer_recursive(self.model, ablation_layer, target_layer)
        return weights
