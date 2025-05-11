import fiftyone as fo
import fiftyone.core.metadata as fom
import fiftyone.brain as fob
import numpy as np
import pathlib
from pathlib import Path
import numpy as np

# 1) Monkey‑patch: todo PosixPath vira WindowsPath
# pathlib.PosixPath = pathlib.WindowsPath
# Carrega dados


def load_embeddings(embeddings_path):
    data = np.load(embeddings_path, allow_pickle=True)

    class_mapping = data['class_mapping'].item()

    db_embeddings = data['db_embeddings']
    db_paths = data['db_path']
    db_labels = np.array([class_mapping[i] for i in data['db_labels']])

    q_embeddings = data['query_embeddings']
    q_paths = data['query_paths']
    q_labels = np.array([class_mapping[i] for i in data['query_labels']])

    return [
        (db_embeddings, db_labels, db_paths, 'db'),
        (q_embeddings, q_labels, q_paths, 'query'),
    ]


if __name__ == '__main__':
    embeddings_list = [
        (
            'artifacts/glomerulo/embeddings_uni/embeddings_2025-04-16_04-16-09.npz',
            'glomerulo_uni',
        ),
        (
            'artifacts/glomerulo/embeddings_uni_fsl/embeddings_2025-04-20_15-19-51.npz',
            'glomerulo_uni_fsl',
        ),
        (
            'notebooks/all_data_embeddings_uni_fsl.npz',
            'all_data_glomerulo_uni_fsl',
        ),
        (
            'artifacts/bracs-resized/embeddings_uni/embeddings_2025-04-16_03-05-03.npz',
            'bracs',
        ),
        (
            'artifacts/bracs-resized/embeddings_uni_fsl/embeddings_2025-04-20_15-02-15.npz',
            'bracs_uni_fsl',
        ),
        (
            'artifacts/CRC-VAL-HE-7K-splitted/embeddings_uni/embeddings_2025-04-16_03-33-52.npz',
            'CRC-VAL-HE-7K',
        ),
        (
            'artifacts/ovarian-cancer-splitted/embeddings_uni/embeddings_2025-04-16_04-35-27.npz',
            'ovarian-cancer',
        ),
        (
            'artifacts/skin-cancer-splitted/embeddings_uni/embeddings_2025-04-16_04-58-53.npz',
            'skin-cancer',
        ),
    ]

    # Carrega e cria um dataset para cada arquivo de embedding
    for i, (emb_path, dataset_name) in enumerate(embeddings_list):
        # Nome único para cada dataset
        # dataset_name = f"embeddings_{dataset_name}_{i+1}"
        for embeddings, labels, paths, emb_type in load_embeddings(emb_path):
            # Cria um dataset para cada tipo de embedding

            dataset_name = f'{dataset_name}_{emb_type}'

            # Deleta se já existir
            if fo.dataset_exists(dataset_name):
                continue
                # fo.delete_dataset(dataset_name)

            dataset = fo.Dataset(dataset_name)

            # Adiciona samples
            for idx, (path, label, emb) in enumerate(
                zip(paths, labels, embeddings)
            ):
                sample = fo.Sample(filepath=path)
                sample['embedding'] = emb.tolist()
                sample['label'] = fo.Classification(label=label)
                sample['sample_id'] = f'{i+1}_{idx}'
                dataset.add_sample(sample)

            dataset.save()

            # Indexa os embeddings para navegação por similaridade
            fob.compute_visualization(
                dataset,
                embeddings='embedding',
                brain_key='embedding_viz',
                method='umap',
                num_dims=2,
            )

            print(
                f"✅ Dataset '{dataset_name}' criado com {len(dataset)} amostras"
            )

    # Lança o app com o último dataset carregado (ou escolha manualmente)
    session = fo.launch_app(dataset, port=5151)
    session.wait()
