import time
import json
import os
from pathlib import Path
import sys




sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from metrics.metric_base import MetricBase
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity



class MapAtK(MetricBase):
    def __init__(self, k_values, similarity_fn=(cosine_similarity, "similarity"), **kwargs):
        """
        Initialize the MapAtK metric with a list of k values.
        
        Args:
            k_values (list): List of integers representing the k values to compute Accuracy for.
            similarity_fn (tuple): A tuple containing:
                - A function to compute similarity or distance between embeddings.
                - A string indicating the type of function: "similarity" or "distance".
                Default is (cosine_similarity, "similarity").
            **kwargs: Additional properties that are not explicitly required by this class.
        """
        if not k_values:
            raise ValueError("k_values must be provided.")
        
        self.k_values = sorted(k_values)
        
        if not isinstance(similarity_fn, tuple) or len(similarity_fn) != 2:
            raise ValueError("similarity_fn must be a tuple of (function, type_string)")
        
        self.sim_function, self.sim_type = similarity_fn
        if self.sim_type not in ["similarity", "distance"]:
            raise ValueError("similarity_fn type must be either 'similarity' or 'distance'")

    def map_at_k(self, embeddings_dict, k_total, is_last=False):
        """
        Calculate Mean Average Precision at k for image retrieval.
        
        Parameters:
        -----------
        embeddings_dict : dict
            Dictionary containing the following keys:
            - 'query_embeddings': Embeddings of query images
            - 'query_labels': Labels of query images
            - 'query_classes': Class names of query images
            - 'query_paths': Paths to query images
            - 'db_embeddings': Embeddings of database images
            - 'db_labels': Labels of database images
            - 'db_path': Paths to database images (new key)
            - 'class_mapping': Dictionary mapping labels to class names
        k_total : int
            Number of retrievals to consider
        is_last : bool, optional
            Whether this is the last k value to evaluate (determines whether to return full results)
            
        Returns:
        --------
        float
            MAP@k value
        dict or None
            Full retrieval results if is_last is True, None otherwise
        """
        # Extract data from embeddings dictionary
        query_embeddings = embeddings_dict['query_embeddings']
        query_labels = embeddings_dict['query_labels']
        query_classes = embeddings_dict['query_classes']
        query_paths = embeddings_dict['query_paths']
        database_embeddings = embeddings_dict['db_embeddings']
        database_labels = embeddings_dict['db_labels']
        database_paths = embeddings_dict['db_path']
        class_mapping = embeddings_dict['class_mapping'] if isinstance(embeddings_dict['class_mapping'], dict) else embeddings_dict['class_mapping'].item()

        # Calculate similarity matrix
        similarity_matrix = self.sim_function(query_embeddings, database_embeddings)
        
        # Adjust sorting based on similarity or distance type
        if self.sim_type == "distance":
            # For distance metrics (lower is better), sort in ascending order
            sorted_indices = np.argsort(similarity_matrix, axis=1)[:, :k_total]
        else:  # similarity
            # For similarity metrics (higher is better), sort in descending order
            sorted_indices = np.argsort(-similarity_matrix, axis=1)[:, :k_total]
            
        num_queries = similarity_matrix.shape[0]
        query_retrievals = []
        avg_precisions = []
        map_results = None  # Results to be returned if is_last is True (k_total is the last k value)

        for idx_query in range(num_queries):
            q_label = query_labels[idx_query]
            q_class = query_classes[idx_query]
            q_path = str(query_paths[idx_query])
            q_sims = similarity_matrix[idx_query]
            sorted_indices_for_query = sorted_indices[idx_query]
            
            relevant_count = 0
            cum_precision = 0.0
            retrieved = []
            
            for k, retrieved_idx in enumerate(sorted_indices_for_query, start=1):
                retrieved_label = database_labels[retrieved_idx]
                retrieved_path = str(database_paths[retrieved_idx]) if database_paths[retrieved_idx] else None
                is_relevant = int(retrieved_label == q_label)
                
                if is_relevant:
                    relevant_count += 1
                    cum_precision += relevant_count / k
                    
                retrieved.append({
                    "k": k,
                    "retrieved_label": int(retrieved_label),
                    "retrieved_class": class_mapping[retrieved_label] if class_mapping else None,
                    "retrieved_path": retrieved_path,
                    "is_relevant": is_relevant,
                    "similarity": float(q_sims[retrieved_idx])
                })
                
            average_precision = cum_precision / relevant_count if relevant_count > 0 else 0.0
            avg_precisions.append(average_precision)
            
            query_retrievals.append({
                "average_precision": average_precision,
                "query_label": int(q_label),
                "query_class": q_class,
                "query_path": q_path,
                "retrieved": retrieved,
            })
            
            # print(f"Query {idx_query}: Average Precision = {average_precision:.4f}")
            
        mapk = float(np.mean(avg_precisions))
        
        if is_last:
            map_results = {"mapk": mapk, "k": k_total, "query_retrievals": query_retrievals}
            
        return mapk, map_results

    def __call__(self, model=None, train_loader=None, test_loader=None, embeddings=None, config=None, logger=None):
        """
        - model: The trained model to use for generating embeddings
        - train_loader: DataLoader for the training dataset
        - test_loader: DataLoader for the testing dataset
        - embeddings: Precomputed embeddings for evaluation
        - config: Configuration object (optional)
        - logger: Logger instance for logging messages
        """
        if embeddings is None:
            raise ValueError("Embeddings must be provided.")
            
        logger.info("Computing MAP@K...")
        map_results = {}
        query_retrievals = None

        # Calculate MAP@k for each k value
        for k in self.k_values:
            is_last = k == max(self.k_values)
            map_results[f'mapAt{k}'], query_retrievals_at_k = self.map_at_k(
                embeddings, k, is_last
            )
            _map_at_k = map_results[f'mapAt{k}']
            logger.info(f"MAP@{k}: {_map_at_k:.4f}")
            
        # Store the last query retrievals
        if query_retrievals_at_k:
            query_retrievals = query_retrievals_at_k
            
        return {
            "map_at_k_results": map_results,
            "map_at_k_query_details": query_retrievals
        }
        
        

if __name__ == '__main__':
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from torch.utils.data import DataLoader
    from dataloaders.dataset_terumo import TerumoImageDataset
    import torchvision.models as models
    import torch.nn as nn

    import tqdm

    class SimpleLogger:
        def info(self, metrics):
            print(metrics)

    num_classes = 6
    model = models.resnet50(pretrained=True)
    model.fc = nn.Identity()

    root_dir = './datasets/final/glomerulo/train'
    custom_mapping = {
        'Crescent': 0,
        'Hypercellularity': 1,
        'Membranous': 2,
        'Normal': 3,
        'Podocytopathy': 4,
        'Sclerosis': 5,
    }

    # Define transformations using Albumentations
    data_transforms = A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ]
    )

    # Create the dataset
    dataset = TerumoImageDataset(
        root_dir=root_dir,
        transform=data_transforms,
        class_mapping=custom_mapping,
    )
    dataset_test = TerumoImageDataset(
        root_dir=root_dir.replace('train', 'test'),
        transform=data_transforms,
        class_mapping=custom_mapping,
    )

    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=3,
        pin_memory=True,
    )

    test_loader = DataLoader(
        dataset_test,
        batch_size=32,
        shuffle=False,
        num_workers=3,
        pin_memory=True,
    )
    

    # for k in [1, 5, 10]:
    #     map_at_k = MapAtK(k)
    #     metrics = map_at_k(model, train_loader, train_loader, None, simple_logger)
    #     print(metrics)

    # map_at_k = MapAtK([1, 5, 10])
    # embeddings = np.load('artifacts/embeddings_dino_2025-04-12_02-56-06.npz', allow_pickle=True)
    # metrics = map_at_k(model, train_loader, test_loader, embeddings, None, SimpleLogger())
    # print(metrics)
