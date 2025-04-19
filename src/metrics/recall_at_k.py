import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from metrics.metric_base import MetricBase
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics.pairwise import cosine_similarity
import tqdm


class RecallAtK(MetricBase):
    def __init__(
        self,
        k_values=None,
        similarity_fn=(cosine_similarity, 'similarity'),
        **kwargs,
    ):
        """
        Initialize the RecallAtK metric with a list of k values and a similarity function.

        Args:
            k_values (list): List of integers representing the k values to compute Recall for.
            similarity_fn (tuple): A tuple containing:
                - A function to compute similarity or distance between embeddings.
                - A string indicating the type of function: "similarity" or "distance".
                Default is (cosine_similarity, "similarity").
            **kwargs: Additional properties that are not explicitly required by this class.
        """
        # Explicit required properties
        if k_values is None:
            raise ValueError('k_values must be provided.')

        self.k_values = sorted(k_values)

        if not isinstance(similarity_fn, tuple) or len(similarity_fn) != 2:
            raise ValueError(
                'similarity_fn must be a tuple of (function, type_string)'
            )

        self.sim_function, self.sim_type = similarity_fn
        if self.sim_type not in ['similarity', 'distance']:
            raise ValueError(
                "similarity_fn type must be either 'similarity' or 'distance'"
            )

    def compute_recall_at_k(self, embeddings_dict, k, is_last=False):
        """
        Compute Recall@K for the given queries and database.

        Args:
            embeddings_dict : dict
                Dictionary containing the following keys:
                - 'query_embeddings': Embeddings of query images
                - 'query_labels': Labels of query images
                - 'query_classes': Class names of query images (optional)
                - 'query_paths': Paths to query images (optional)
                - 'db_embeddings': Embeddings of database images
                - 'db_labels': Labels of database images
                - 'db_path': Paths to database images (optional)
                - 'class_mapping': Dictionary mapping labels to class names (optional)
            k: The value of K for Recall@K.
            is_last: bool, optional
                Whether this is the last k value to evaluate (determines whether to return full results)

        Returns:
            float: The Recall@K score.
            dict or None: Additional result details if is_last is True, None otherwise
        """
        # Extract data from embeddings dictionary
        query_embeddings = embeddings_dict['query_embeddings']
        query_labels = embeddings_dict['query_labels']
        database_embeddings = embeddings_dict['db_embeddings']
        database_labels = embeddings_dict['db_labels']

        # Optional fields with defaults
        query_classes = embeddings_dict.get('query_classes', None)
        query_paths = embeddings_dict.get('query_paths', None)
        database_paths = embeddings_dict.get('db_path', None)
        class_mapping = embeddings_dict.get('class_mapping', None)

        # Calculate similarity matrix
        similarity_matrix = self.sim_function(
            query_embeddings, database_embeddings
        )

        # Determine sorting order based on similarity type
        if self.sim_type == 'distance':
            # For distance metrics (lower is better), sort in ascending order
            sorted_indices = np.argsort(similarity_matrix, axis=1)
        else:  # similarity
            # For similarity metrics (higher is better), sort in descending order
            sorted_indices = np.argsort(-similarity_matrix, axis=1)

        num_queries = query_embeddings.shape[0]
        recalls = []
        query_details = []

        for i in range(num_queries):
            query_label = query_labels[i]
            similarities = similarity_matrix[i]
            sorted_indices_for_query = sorted_indices[i]

            # Optional: get query class and path if available
            query_class = (
                query_classes[i] if query_classes is not None else None
            )
            query_path = (
                str(query_paths[i]) if query_paths is not None else None
            )

            # Get the top K retrieval indices
            top_k_indices = sorted_indices_for_query[:k]

            # Count relevant items in the top K
            relevant_items = []
            relevant_count = 0

            for idx_k, idx in enumerate(top_k_indices, start=1):
                retrieved_label = database_labels[idx]
                is_relevant = int(retrieved_label == query_label)
                if is_relevant:
                    relevant_count += 1

                # Get additional information if available
                retrieved_class = None
                if (
                    class_mapping is not None
                    and retrieved_label in class_mapping
                ):
                    retrieved_class = class_mapping[retrieved_label]

                retrieved_path = None
                if database_paths is not None:
                    retrieved_path = (
                        str(database_paths[idx])
                        if database_paths[idx]
                        else None
                    )

                # Store details for each retrieved item
                retrieved_item = {
                    'k': idx_k,
                    'retrieved_label': int(retrieved_label),
                    'retrieved_class': retrieved_class,
                    'retrieved_path': retrieved_path,
                    'is_relevant': is_relevant,
                    'similarity': float(similarities[idx]),
                }

                # Add optional fields if available
                if retrieved_class is not None:
                    retrieved_item['retrieved_class'] = retrieved_class
                if retrieved_path is not None:
                    retrieved_item['retrieved_path'] = retrieved_path

                relevant_items.append(retrieved_item)

            # Count total relevant items in the database
            total_relevant = (database_labels == query_label).sum()

            # Calculate recall
            recall = (
                relevant_count / total_relevant if total_relevant > 0 else 0.0
            )
            recalls.append(recall)

            # Create query detail entry
            query_detail = {
                'recall': recall,
                'query_label': int(query_label),
                'query_class': query_class,
                'query_path': query_path,
                'relevant_count': int(relevant_count),
                'total_relevant': int(total_relevant),
                'retrieved': relevant_items,
            }

            query_details.append(query_detail)

        recall_at_k = float(np.mean(recalls))

        # Return results based on whether this is the last k value
        if is_last:
            results = {
                'recall_at_k': recall_at_k,
                'k': k,
                'query_details': query_details,
            }
            return recall_at_k, results
        else:
            return recall_at_k, None

    def __call__(
        self,
        model=None,
        train_loader=None,
        test_loader=None,
        embeddings=None,
        config=None,
        logger=None,
    ):
        """
        Compute the Recall@K for the given model and dataset.

        Args:
            model: The model to evaluate.
            train_loader: The data loader for training data (used for creating the database).
            test_loader: The data loader for test data (used for queries).
            embeddings: Precomputed embeddings for evaluation.
            config: Configuration object.
            logger: Logger object for logging information.

        Returns:
            dict: A dictionary containing Recall@K values for each k in self.k_values.
        """
        if embeddings is None:
            raise ValueError('Embeddings must be provided.')

        # Compute Recall for all k values
        logger.info('Computing Recall@K...')
        recall_results = {}
        query_details = None

        for k in self.k_values:
            is_last = k == max(self.k_values)

            recall_value, details = self.compute_recall_at_k(
                embeddings, k, is_last
            )

            recall_results[f'recallAt{k}'] = recall_value
            logger.info(f'Recall@{k}: {recall_value:.4f}')

            # Store query details for the last k value
            if is_last and details is not None:
                query_details = details

        return {
            'recall_at_k_results': recall_results,
            'recall_at_k_query_details': query_details,
        }
