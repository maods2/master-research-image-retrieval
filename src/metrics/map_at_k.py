import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pipelines.train_pipes.multilabel_train import load_pretrained_model
from metrics.metric_base import MetricBase
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tqdm


class MapAtK(MetricBase):
    def __init__(
        self, k_values=None, similarity_fn=cosine_similarity, **kwargs
    ):
        """
        Initialize the MapAtK metric with a list of k values.

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

    def compute_map_at_k(
        self,
        query_embeddings,
        query_labels,
        database_embeddings,
        database_labels,
        k,
    ):
        """
        Compute MAP@K for the given queries and database.

        Args:
            query_embeddings: Embeddings of query images.
            query_labels: Labels of query images.
            database_embeddings: Embeddings of the database.
            database_labels: Labels of the database.
            k: The value of K for MAP@K.

        Returns:
            float: The MAP@K score.
        """
        # Use the dynamically provided similarity/distance function
        similarity_matrix = self.similarity_fn(
            query_embeddings, database_embeddings
        )

        num_queries = similarity_matrix.shape[0]
        average_precisions = []

        for i in range(num_queries):
            query_label = query_labels[i]
            similarities = similarity_matrix[i]
            sorted_indices = np.argsort(-similarities)  # Descending order

            # Compute precision at K
            relevant_count = 0
            precision_at_k = 0.0
            for rank, idx in enumerate(sorted_indices[:k], start=1):
                if database_labels[idx] == query_label:
                    relevant_count += 1
                    precision_at_k += relevant_count / rank

            if relevant_count > 0:
                average_precisions.append(
                    precision_at_k / min(k, relevant_count)
                )
            else:
                average_precisions.append(0.0)

        return np.mean(average_precisions)

    def __call__(self, model, train_loader, test_loader, config, logger):
        """
        Compute the MAP@K for the given model and dataset.

        Args:
            model: The model to evaluate.
            train_loader: The data loader for training data (used for creating the database).
            test_loader: The data loader for test data (used for queries).
            config: Configuration object.
            logger: Logger object for logging information.

        Returns:
            dict: A dictionary containing MAP@K values for each k in self.k_values.
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Create embeddings database from the training data
        logger.info('Creating embeddings database from training data...')
        database_embeddings, database_labels = self.create_embeddings(
            model, train_loader, device, logger, desc='Creating database'
        )

        # Generate query embeddings and labels from the test data
        logger.info('Generating query embeddings from test data...')
        query_embeddings, query_labels = self.create_embeddings(
            model, test_loader, device, logger, desc='Generating queries'
        )

        # Compute MAP for all k values
        logger.info('Computing MAP@K...')
        map_results = {}
        for k in self.k_values:
            map_results[f'mapAt{k}'] = self.compute_map_at_k(
                query_embeddings,
                query_labels,
                database_embeddings,
                database_labels,
                k,
            )
            logger.info(f"MAP@{k}: {map_results[f'mapAt{k}']}")

        return map_results


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

    root_dir = '../../datasets/final/terumo/train'
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
        shuffle=True,
        num_workers=3,
        pin_memory=True,
    )

    test_loader = DataLoader(
        dataset_test,
        batch_size=32,
        shuffle=True,
        num_workers=3,
        pin_memory=True,
    )

    # for k in [1, 5, 10]:
    #     map_at_k = MapAtK(k)
    #     metrics = map_at_k(model, train_loader, train_loader, None, simple_logger)
    #     print(metrics)

    map_at_k = MapAtK([1, 5, 10])
    metrics = map_at_k(model, train_loader, test_loader, None, SimpleLogger())
    print(metrics)
