"""
See original implementation at
https://github.com/floodsung/LearningToCompare_FSL
"""
from typing import Optional

import torch
from torch import Tensor, nn

from easyfsl.modules.predesigned_modules import default_relation_module

from .few_shot_classifier import FewShotClassifier
from .utils import compute_prototypes


class RelationNetworks(FewShotClassifier):
    """
    Sung, Flood, Yongxin Yang, Li Zhang, Tao Xiang, Philip HS Torr, and Timothy M. Hospedales.
    "Learning to compare: Relation network for few-shot learning." (2018)
    https://openaccess.thecvf.com/content_cvpr_2018/papers/Sung_Learning_to_Compare_CVPR_2018_paper.pdf

    In the Relation Networks algorithm, we first extract feature maps for both support and query
    images. Then we compute the mean of support features for each class (called prototypes).
    To predict the label of a query image, its feature map is concatenated with each class prototype
    and fed into a relation module, i.e. a CNN that outputs a relation score. Finally, the
    classification vector of the query is its relation score to each class prototype.

    Note that for most other few-shot algorithms we talk about feature vectors, because for each
    input image, the backbone outputs a 1-dim feature vector. Here we talk about feature maps,
    because for each input image, the backbone outputs a "feature map" of shape
    (n_channels, width, height). This raises different constraints on the architecture of the
    backbone: while other algorithms require a "flatten" operation in the backbone, here "flatten"
    operations are forbidden.

    Relation Networks use Mean Square Error. This is unusual because this is a classification
    problem. The authors justify this choice by the fact that the output of the model is a relation
    score, which makes it a regression problem. See the article for more details.
    """

    def __init__(
        self,
        *args,
        feature_dimension: int,
        relation_module: Optional[nn.Module] = None,
        **kwargs,
    ):
        """
        Build Relation Networks by calling the constructor of FewShotClassifier.
        Args:
            feature_dimension: first dimension of the feature maps extracted by the backbone.
            relation_module: module that will take the concatenation of a query features vector
                and a prototype to output a relation score. If none is specific, we use the default
                relation module from the original paper.
        """
        super().__init__(*args, **kwargs)

        self.feature_dimension = feature_dimension

        # Here we build the relation module that will output the relation score for each
        # (query, prototype) pair. See the function docstring for more details.
        self.relation_module = (
            relation_module
            if relation_module
            else default_relation_module(self.feature_dimension)
        )

    def process_support_set(
        self,
        support_images: Tensor,
        support_labels: Tensor,
    ):
        """
        Overrides process_support_set of FewShotClassifier.
        Extract feature maps from the support set and store class prototypes.
        """

        support_features = self.compute_features(support_images)
        self._validate_features_shape(support_features)
        self.prototypes = compute_prototypes(support_features, support_labels)

    def forward(self, query_images: Tensor) -> Tensor:
        """
        Overrides method forward in FewShotClassifier.
        Predict the label of a query image by concatenating its feature map with each class
        prototype and feeding the result into a relation module, i.e. a CNN that outputs a relation
        score. Finally, the classification vector of the query is its relation score to each class
        prototype.
        """
        query_features = self.compute_features(query_images)
        self._validate_features_shape(query_features)

        # For each pair (query, prototype), we compute the concatenation of their feature maps
        # Given that query_features is of shape (n_queries, n_channels, width, height), the
        # constructed tensor is of shape (n_queries * n_prototypes, 2 * n_channels, width, height)
        # (2 * n_channels because prototypes and queries are concatenated)
        query_prototype_feature_pairs = torch.cat(
            (
                self.prototypes.unsqueeze(dim=0).expand(
                    query_features.shape[0], -1, -1, -1, -1
                ),
                query_features.unsqueeze(dim=1).expand(
                    -1, self.prototypes.shape[0], -1, -1, -1
                ),
            ),
            dim=2,
        ).view(-1, 2 * self.feature_dimension, *query_features.shape[2:])

        # Each pair (query, prototype) is assigned a relation scores in [0,1]. Then we reshape the
        # tensor so that relation_scores is of shape (n_queries, n_prototypes).
        relation_scores = self.relation_module(query_prototype_feature_pairs).view(
            -1, self.prototypes.shape[0]
        )

        return self.softmax_if_specified(relation_scores, temperature=self.temperature)

    def _validate_features_shape(self, features):
        if len(features.shape) != 4:
            raise ValueError(
                "Illegal backbone for Relation Networks. "
                "Expected output for an image is a 3-dim  tensor of shape (n_channels, width, height)."
            )
        if features.shape[1] != self.feature_dimension:
            raise ValueError(
                f"Expected feature dimension is {self.feature_dimension}, but got {features.shape[1]}."
            )

    @staticmethod
    def is_transductive() -> bool:
        return False
