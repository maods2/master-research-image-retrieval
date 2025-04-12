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
        self, k_values=None, similarity_fn=(cosine_similarity, "similarity"), **kwargs
    ):
        """
        Initialize the PrecisionAtK metric with a list of k values and a similarity function.

        Args:
            k_values (list): List of integers representing the k values to compute Accuracy for.
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
            raise ValueError("similarity_fn must be a tuple of (function, type_string)")
        
        self.sim_function, self.sim_type = similarity_fn
        if self.sim_type not in ["similarity", "distance"]:
            raise ValueError("similarity_fn type must be either 'similarity' or 'distance'")
            


    def compute_precision_at_k(self, embeddings_dict, k, is_last=False):
        """
        Compute Precision@K for the given queries and database.

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
            k: The value of K for Precision@K.
            is_last: bool, optional
                Whether this is the last k value to evaluate (determines whether to return full results)

        Returns:
            float: The Precision@K score.
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
        similarity_matrix = self.sim_function(query_embeddings, database_embeddings)

        # Determine sorting order based on similarity type
        if self.sim_type == "distance":
            # For distance metrics (lower is better), sort in ascending order
            sorted_indices = np.argsort(similarity_matrix, axis=1)
        else:  # similarity
            # For similarity metrics (higher is better), sort in descending order
            sorted_indices = np.argsort(-similarity_matrix, axis=1)

        num_queries = query_embeddings.shape[0]
        precisions = []
        query_details = []

        map_results = None
        for i in range(num_queries):
            query_label = query_labels[i]
            similarities = similarity_matrix[i]
            sorted_indices_for_query = sorted_indices[i]
            
            # Optional: get query class and path if available
            query_class = query_classes[i] if query_classes is not None else None
            query_path = str(query_paths[i]) if query_paths is not None else None
            
            # Take top k indices
            top_k_indices = sorted_indices_for_query[:k]
            
            # Compute precision at K
            relevant_count = 0
            retrieved_items = []
            
            for idx_k, idx in enumerate(top_k_indices, start=1):
                retrieved_label = database_labels[idx]
                is_relevant = int(retrieved_label == query_label)
                if is_relevant:
                    relevant_count += 1
                
                # Get additional information if available
                retrieved_class = None
                if class_mapping is not None and retrieved_label in class_mapping:
                    retrieved_class = class_mapping[retrieved_label]
                    
                retrieved_path = None
                if database_paths is not None:
                    retrieved_path = str(database_paths[idx]) if database_paths[idx] else None
                
                # Store details for each retrieved item
                retrieved_item = {
                    "k": idx_k,
                    "retrieved_label": int(retrieved_label),
                    "retrieved_class": retrieved_class,
                    "retrieved_path": retrieved_path,
                    "is_relevant": is_relevant,
                    "similarity": float(similarities[idx])
                }
                
                # Add optional fields if available
                if retrieved_class is not None:
                    retrieved_item["retrieved_class"] = retrieved_class
                if retrieved_path is not None:
                    retrieved_item["retrieved_path"] = retrieved_path
                    
                retrieved_items.append(retrieved_item)

            precision = relevant_count / k if k > 0 else 0.0
            precisions.append(precision)
            
            # Create query detail entry
            query_detail = {
                "precision": precision,
                "query_label": int(query_label),
                "query_class": query_class,
                "query_path": query_path,
                "retrieved": retrieved_items
            }
            
            # Add optional fields if available
            if query_class is not None:
                query_detail["query_class"] = query_class
            if query_path is not None:
                query_detail["query_path"] = query_path
                
            query_details.append(query_detail)

        precision_at_k = float(np.mean(precisions))
        
        # Return results based on whether this is the last k value
        if is_last:
            map_results = {
                "precision_at_k": precision_at_k,
                "k": k,
                "query_details": query_details
            }
            
        return precision_at_k, map_results
        


    def __call__(self, model=None, train_loader=None, test_loader=None, embeddings=None, config=None, logger=None):
        """
        Compute the Precision@K for the given model and dataset.

        Args:
            model: The model to evaluate.
            train_loader: The data loader for training data (used for creating the database).
            test_loader: The data loader for test data (used for queries).
            embeddings: Precomputed embeddings for evaluation.
            config: Configuration object.
            logger: Logger object for logging information.

        Returns:
            dict: A dictionary containing Precision@K values for each k in self.k_values.
        """
        if embeddings is None:
            raise ValueError("Embeddings must be provided.")
            
        # Compute Precision for all k values
        logger.info('Computing Precision@K...')
        precision_results = {}
        query_details = None
        
        for k in self.k_values:
            is_last = k == max(self.k_values)
            
            precision_value, details = self.compute_precision_at_k(
                embeddings,
                k,
                is_last
            )
            
            precision_results[f'precisionAt{k}'] = precision_value
            logger.info(f"Precision@{k}: {precision_value:.4f}")
            
            # Store query details for the last k value
            if is_last and details is not None:
                query_details = details

        return {
            "precision_at_k_results": precision_results,
            "precision_at_k_query_details": query_details
        }
