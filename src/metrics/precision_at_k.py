import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from metrics.metric_base import MetricBase
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics.pairwise import cosine_similarity
import tqdm


class PrecisionAtK(MetricBase):
    def __init__(
        self, k_values=None, similarity_fn=cosine_similarity, **kwargs
    ):
        """
        Initialize the PrecisionAtK metric with a list of k values and a similarity function.

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
        self.database = {}
        self.query = {}

    def create_embeddings(
        self, model, data_loader, device, logger, desc='Extracting features'
    ):
        """
        Generate embeddings and labels from a given data loader.

        Args:
            model: The model used for generating embeddings.
            data_loader: The data loader containing the data.
            device: The device to perform computations (e.g., 'cuda' or 'cpu').
            logger: Logger object for logging information.
            desc: Description for the progress bar.

        Returns:
            tuple: A tuple containing embeddings (np.ndarray) and labels (np.ndarray).
        """
        embeddings = []
        labels = []
        model.eval()
        model.to(device)
        data_loader = tqdm.tqdm(data_loader, desc=desc)

        for img, label in data_loader:
            with torch.no_grad():
                embedding = model(img.to(device))
            embeddings.append(embedding.cpu().numpy())
            labels.append(label.argmax(dim=1).cpu().numpy())

        embeddings = np.concatenate(embeddings, axis=0)
        labels = np.concatenate(labels, axis=0)

        logger.info(
            f'Embeddings shape: {embeddings.shape}, Labels shape: {labels.shape}'
        )
        return embeddings, labels

    def compute_precision_at_k(
        self,
        query_embeddings,
        query_labels,
        database_embeddings,
        database_labels,
        k,
    ):
        """
        Compute Precision@K for the given queries and database.

        Args:
            query_embeddings: Embeddings of query images.
            query_labels: Labels of query images.
            database_embeddings: Embeddings of the database.
            database_labels: Labels of the database.
            k: The value of K for Precision@K.

        Returns:
            float: The Precision@K score.
        """
        # Use the dynamically provided similarity/distance function
        similarity_matrix = self.similarity_fn(
            query_embeddings, database_embeddings
        )

        # Check if similarity or distance is being calculated
        if (
            np.min(similarity_matrix) >= 0
        ):  # Similarity should be non-negative (e.g., cosine)
            similarity_matrix = (
                -similarity_matrix
            )  # Invert to make higher values more similar
        elif (
            np.min(similarity_matrix) < 0
        ):  # In the case of distance (e.g., Euclidean)
            # We assume the function returns a distance and the smaller value is closer
            # No need to invert the matrix, as smaller values are already more similar
            pass
        else:
            raise ValueError('Similarity/distance matrix is not valid')

        num_queries = similarity_matrix.shape[0]
        precisions = []

        for i in range(num_queries):
            query_label = query_labels[i]
            similarities = similarity_matrix[i]
            sorted_indices = np.argsort(
                -similarities
            )  # Descending order (higher similarity first)

            # Compute precision at K
            relevant_count = 0
            for idx in sorted_indices[:k]:
                if database_labels[idx] == query_label:
                    relevant_count += 1

            precision = relevant_count / k
            precisions.append(precision)

        return np.mean(precisions)

    def __call__(self, model, train_loader, test_loader, config, logger):
        """
        Compute the Precision@K for the given model and dataset.

        Args:
            model: The model to evaluate.
            train_loader: The data loader for training data (used for creating the database).
            test_loader: The data loader for test data (used for queries).
            config: Configuration object.
            logger: Logger object for logging information.

        Returns:
            dict: A dictionary containing Precision@K values for each k in self.k_values.
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if not 'embeddings' in self.database:
            # Create embeddings database from the training data
            logger.info('Creating embeddings database from training data...')
            (
                self.database['embeddings'],
                self.database['labels'],
            ) = self.create_embeddings(
                model, train_loader, device, logger, desc='Creating database'
            )

            # Generate query embeddings and labels from the test data
            logger.info('Generating query embeddings from test data...')
            (
                self.query['embeddings'],
                self.query['labels'],
            ) = self.create_embeddings(
                model, test_loader, device, logger, desc='Generating queries'
            )

        # Compute Precision for all k values
        logger.info('Computing Precision@K...')
        precision_results = {}
        for k in self.k_values:
            precision_results[f'precisionAt{k}'] = self.compute_precision_at_k(
                self.query['embeddings'],
                self.query['labels'],
                self.database['embeddings'],
                self.database['labels'],
                k,
            )
            logger.info(
                f"Precision@{k}: {precision_results[f'precisionAt{k}']}"
            )

        return precision_results
