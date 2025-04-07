import time
import json
import os
from pathlib import Path
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


class MapAtK2(MetricBase):
    def __init__(self, k_values, similarity_fn=cosine_similarity, output_dir="map_results"):
        if not k_values:
            raise ValueError("k_values must be provided.")
        self.k_values = sorted(k_values)
        self.similarity_fn = similarity_fn
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compute_map_at_k(self, 
                        query_embeddings, query_labels, 
                        database_embeddings, database_labels, 
                        k, 
                        save_details=False, 
                        query_paths=None, 
                        query_classes=None,
                        experiment_name="exp"):
        """
        Se save_details=True e k for o maior de k_values, retorna também a lista de dicionários
        com detalhes de cada query para salvamento.
        """
        sim = self.similarity_fn(query_embeddings, database_embeddings)
        num_q = sim.shape[0]
        avg_precisions = []
        details = []

        for i in range(num_q):
            q_label = query_labels[i]
            sims = sim[i]
            ranked = np.argsort(-sims)[:k]  # índices top-k
            relevant_count = 0
            prec_sum = 0.0

            # Para relatório detalhado
            retrieved = []
            for rank, db_idx in enumerate(ranked, start=1):
                db_label = database_labels[db_idx]
                is_rel = int(db_label == q_label)
                retrieved.append({
                    "rank": rank,
                    "db_label": int(db_label),
                    "is_relevant": is_rel,
                    "similarity": float(sims[db_idx])
                })
                if is_rel:
                    relevant_count += 1
                    prec_sum += relevant_count / rank

            ap = (prec_sum / min(k, relevant_count)) if relevant_count>0 else 0.0
            avg_precisions.append(ap)

            if save_details:
                details.append({
                    "query_id": query_paths[i] if query_paths else int(i),
                    "query_label": int(q_label),
                    "query_class": query_classes[i] if query_classes else None,
                    "average_precision": float(ap),
                    "retrieved": retrieved
                })

        mapk = float(np.mean(avg_precisions))
        return mapk, details

    def __call__(self, 
                 model, 
                 train_loader, 
                 test_loader, 
                 embeddings, 
                 config, 
                 logger,
                 experiment_name="exp"):
        """
        embeddings deve conter:
          embeddings['query_embeddings'], embeddings['query_labels'],
          embeddings['db_embeddings'], embeddings['db_labels'],
          opcional: embeddings['query_ids'], embeddings['db_ids']
        """
        logger.info("Computing MAP@K...")
        results = {}
        max_k = max(self.k_values)
        # extrai ids se existirem
        q_paths = [str(i) for i in test_loader.dataset.image_paths]
        q_classes = [str(i) for i in test_loader.dataset.labels]


        for k in self.k_values:
            save = (k == max_k)
            mapk, details = self.compute_map_at_k(
                embeddings['query_embeddings'],
                embeddings['query_labels'],
                embeddings['db_embeddings'],
                embeddings['db_labels'],
                k,
                save_details=save,
                query_paths=q_paths,
                query_classes=q_classes,
                experiment_name=experiment_name
            )
            results[f"mapAt{k}"] = mapk
            logger.info(f"MAP@{k}: {mapk:.4f}")

            if save:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                base = f"{experiment_name}_MAP{k}_{timestamp}"
                # salva JSON
                json_path = self.output_dir / f"{base}.json"
                with open(json_path, "w") as f:
                    json.dump(details, f, indent=2)
                logger.info(f"Detalhes salvos em JSON: {json_path}")

                

        return results


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

    root_dir = './datasets/final/terumo/train'
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

    map_at_k = MapAtK2([1, 5, 10])
    embeddings = np.load('artifacts/embeddings_clip_2025-04-07_01-31-44.npz', allow_pickle=True)
    metrics = map_at_k(model, train_loader, test_loader,embeddings, None, SimpleLogger())
    print(metrics)
