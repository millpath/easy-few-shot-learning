import torch
from torch import Tensor, nn

from .few_shot_classifier import FewShotClassifier
from easyfsl.methods.utils import compute_prototypes


class Finetune(FewShotClassifier):
    """
    Wei-Yu Chen, Yen-Cheng Liu, Zsolt Kira, Yu-Chiang Frank Wang, Jia-Bin Huang
    A Closer Look at Few-shot Classification (ICLR 2019)
    https://arxiv.org/abs/1904.04232

    Fine-tune prototypes based on classification error on support images.
    Classify queries based on their cosine distances to updated prototypes.
    As is, it is incompatible with episodic training because we freeze the backbone to perform
    fine-tuning.

    This is an inductive method.
    """

    def __init__(
        self,
        *args,
        fine_tuning_steps: int = 200,
        fine_tuning_lr: float = 1e-4,
        temperature: float = 1.0,
        fine_tune_every_prediction: bool = True,
        **kwargs,
    ):
        """
        Args:
            fine_tuning_steps: number of fine-tuning steps
            fine_tuning_lr: learning rate for fine-tuning
            temperature: temperature applied to the logits before computing
                softmax or cross-entropy. Higher temperature means softer predictions.
        """
        super().__init__(*args, **kwargs)
        self.fine_tuning_steps = fine_tuning_steps
        self.fine_tuning_lr = fine_tuning_lr
        self.temperature = temperature
        self.fine_tune_every_prediction = fine_tune_every_prediction
        self.already_fine_tuned = False

    def forward(
        self,
        query_images: Tensor,
    ) -> Tensor:
        """
        Overrides forward method of FewShotClassifier.
        Fine-tune prototypes based on support classification error.
        Then classify w.r.t. to cosine distance to prototypes.
        """

        query_features = self.compute_features(query_images)

        self.performe_fine_tuning()
        
        return self.softmax_if_specified(
            self.cosine_distance_to_prototypes(query_features),
            temperature=self.temperature,
        ).detach()
    
    def performe_fine_tuning(self):
        if self.already_fine_tuned and not self.fine_tune_every_prediction:
            return
        with torch.enable_grad():
            self.prototypes.requires_grad_()
            optimizer = torch.optim.Adam([self.prototypes], lr=self.fine_tuning_lr)
            for _ in range(self.fine_tuning_steps):
                support_logits = self.cosine_distance_to_prototypes(
                    self.support_features
                )
                loss = nn.functional.cross_entropy(
                    self.temperature * support_logits, self.support_labels
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.already_fine_tuned = True

    @staticmethod
    def is_transductive() -> bool:
        return False
    
    def compute_prototypes_and_store_support_set(
        self,
        support_images: Tensor,
        support_labels: Tensor,
    ):
        """
        Extract support features, compute prototypes, and store support labels, features, and prototypes.
        Args:
            support_images: images of the support set of shape (n_support, **image_shape)
            support_labels: labels of support set images of shape (n_support, )
        """
        # Since we fine-tune the prototypes we need to make them leaf variables
        # i.e. we need to freeze the backbone.
        self.backbone.requires_grad_(False)
        self.support_labels = support_labels
        self.support_features = self.compute_features(support_images)
        self._raise_error_if_features_are_multi_dimensional(self.support_features)
        self.prototypes = compute_prototypes(self.support_features, support_labels)
        # enable gradients for backbone to enable regular training
        self.backbone.requires_grad_(True)
