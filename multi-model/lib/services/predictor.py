"""
Prediction service.

Orchestrates prediction using the FG_MFN model and label maps,
converting raw model logits into human-readable attribute predictions.
"""

import logging
from typing import Any, Dict, List

import torch

from lib.models.fg_mfn import FG_MFN

logger = logging.getLogger(__name__)


class Predictor:
    """
    Orchestrates prediction using the model and label maps.

    Takes a loaded FG_MFN model and label maps, runs inference,
    and converts logits to named predictions for each attribute.
    """

    def __init__(
        self,
        model: FG_MFN,
        label_maps: Dict[str, List[str]],
    ) -> None:
        """
        Initialize the Predictor.

        Args:
            model: A loaded FG_MFN model instance in eval mode.
            label_maps: Dictionary mapping attribute names to lists of
                label strings, ordered by class index.

        Raises:
            ValueError: If model or label_maps is None.
        """
        if model is None:
            raise ValueError(
                "model must not be None. Provide a loaded FG_MFN instance."
            )
        if label_maps is None:
            raise ValueError(
                "label_maps must not be None. Provide a dictionary of "
                "attribute names to label lists."
            )

        self.model = model
        self.label_maps = label_maps

    def predict_single(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Predict attributes for a single image.

        Runs the model forward pass and converts logits to named
        predictions for each attribute.

        Args:
            images: Preprocessed image tensor of shape (1, C, H, W).
            input_ids: Token IDs tensor of shape (1, seq_len).
            attention_mask: Attention mask tensor of shape (1, seq_len).

        Returns:
            Dictionary mapping attribute names to their predicted label
            strings, plus confidence scores and numeric class indices.
        """
        with torch.no_grad():
            raw_outputs = self.model(images, input_ids, attention_mask)

        result: Dict[str, Any] = {}
        for attr_name, logits in raw_outputs.items():
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_class = probabilities.max(dim=1)

            predicted_index = predicted_class.item()
            confidence_value = confidence.item()

            label_names = self.label_maps.get(attr_name, [])
            predicted_label_text = (
                label_names[predicted_index]
                if predicted_index < len(label_names)
                else str(predicted_index)
            )

            result[attr_name] = predicted_label_text
            result[f"{attr_name}_confidence"] = confidence_value
            result[f"{attr_name}_predicted_label_num"] = predicted_index

        return result

    def predict_batch(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> List[Dict[str, Any]]:
        """
        Predict attributes for a batch of images.

        Runs the model forward pass on the entire batch and converts
        logits to named predictions for each sample.

        Args:
            images: Batch of image tensors of shape (B, C, H, W).
            input_ids: Batched token IDs of shape (B, seq_len).
            attention_mask: Batched attention mask of shape (B, seq_len).

        Returns:
            List of prediction dictionaries, one per sample.
        """
        with torch.no_grad():
            raw_outputs = self.model(images, input_ids, attention_mask)

        batch_size = images.size(0)
        batch_results: List[Dict[str, Any]] = []

        for sample_index in range(batch_size):
            sample_result: Dict[str, Any] = {}

            for attr_name, logits in raw_outputs.items():
                sample_logits = logits[sample_index].unsqueeze(0)
                probabilities = torch.softmax(sample_logits, dim=1)
                confidence, predicted_class = probabilities.max(dim=1)

                predicted_index = predicted_class.item()
                confidence_value = confidence.item()

                label_names = self.label_maps.get(attr_name, [])
                predicted_label_text = (
                    label_names[predicted_index]
                    if predicted_index < len(label_names)
                    else str(predicted_index)
                )

                sample_result[attr_name] = predicted_label_text
                sample_result[f"{attr_name}_confidence"] = confidence_value
                sample_result[f"{attr_name}_predicted_label_num"] = predicted_index

            batch_results.append(sample_result)

        return batch_results
