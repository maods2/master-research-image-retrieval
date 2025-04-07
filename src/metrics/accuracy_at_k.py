import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pipelines.train_pipes.multilabel_train import load_pretrained_model
from metrics.metric_base import MetricBase
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tqdm


class AccuracyAtK(MetricBase):
    def __init__(
        self, k_values=None, similarity_fn=cosine_similarity, **kwargs
    ):
        """
        Initialize the AccuracyAtK metric with a list of k values and a similarity function.

        Args:
            k_values (list): List of integers representing the k values to compute Accuracy for.
            similarity_fn (callable): A function to compute similarity or distance between embeddings.
                                      Default is cosine similarity.
            **kwargs: Additional properties that are not explicitly required by this class.
        """
        # Explicit required properties
        if k_values is None:
            raise ValueError('k_values must be provided.')
        self.k_values = k_values
        self.similarity_fn = similarity_fn

    def compute_accuracy_at_k(
        self,
        query_embeddings,
        query_labels,
        database_embeddings,
        database_labels,
        k,
    ):
        """
        Compute Accuracy@K for the given queries and database.

        Args:
            query_embeddings: Embeddings of query images.
            query_labels: Labels of query images.
            database_embeddings: Embeddings of the database.
            database_labels: Labels of the database.
            k: The value of K for Accuracy@K.

        Returns:
            float: The Accuracy@K score.
        """
        # Use the dynamically provided similarity/distance function
        similarity_matrix = self.similarity_fn(
            query_embeddings, database_embeddings
        )

        # If the similarity function computes distance, invert it for ranking
        if (
            np.min(similarity_matrix) >= 0
        ):  # Assumption: similarity is always positive
            similarity_matrix = -similarity_matrix

        num_queries = similarity_matrix.shape[0]
        correct_at_k = 0

        for i in range(num_queries):
            query_label = query_labels[i]
            similarities = similarity_matrix[i]
            sorted_indices = np.argsort(
                -similarities
            )  # Descending order (higher similarity first)

            # Get the labels of the top K results
            top_k_labels = database_labels[sorted_indices[:k]]

            # Check if the query label is in the top K
            if query_label in top_k_labels:
                correct_at_k += 1

        # Accuracy is the fraction of queries with at least one correct result in top K
        return correct_at_k / num_queries

    def __call__(self, model, train_loader, test_loader, embeddings, config, logger):
        """
        Compute the Accuracy@K for the given model and dataset.

        Args:
            model: The model to evaluate.
            train_loader: The data loader for training data (used for creating the database).
            test_loader: The data loader for test data (used for queries).
            config: Configuration object.
            logger: Logger object for logging information.

        Returns:
            dict: A dictionary containing Accuracy@K values for each k in self.k_values.
        """
        # Compute Accuracy for all k values
        logger.info('Computing Accuracy@K...')
        accuracy_results = {}
        for k in self.k_values:
            accuracy_results[f'accuracyAt{k}'] = self.compute_accuracy_at_k(
                embeddings['query_embeddings'],
                embeddings['query_labels'],
                embeddings['db_embeddings'],
                embeddings['db_labels'],
                k,
            )
            logger.info(f"Accuracy@{k}: {accuracy_results[f'accuracyAt{k}']}")

        return accuracy_results
