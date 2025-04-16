import fiftyone as fo
import fiftyone.core.dataset as fod
import fiftyone.brain as fob
import numpy as np

# --- Dados de entrada ---
image_paths = [...]  # todos os paths
embeddings = np.array([...])
retrieval_results = {...}  # dict {query_path: [retrieved_path1, retrieved_path2, ...]}
relevance = {...}  # opcional: dict {retrieved_path: True/False}

# --- Criação do dataset ---
dataset = fod.Dataset(name="cbir-evaluation")
samples_map = {}

# Adiciona todas as imagens
for img_path in image_paths:
    sample = fo.Sample(filepath=img_path)
    samples_map[img_path] = sample
    dataset.add_sample(sample)

# Marca tags de query e retrieved
for query_path, retrieved_paths in retrieval_results.items():
    if query_path in samples_map:
        samples_map[query_path].tags.append("query")

    for ret_path in retrieved_paths:
        if ret_path in samples_map:
            samples_map[ret_path].tags.append("retrieved")
            if ret_path in relevance:
                samples_map[ret_path]["relevant"] = relevance[ret_path]

# Salva alterações
dataset.save()

# --- Visualização com embeddings (UMAP, PCA ou t-SNE) ---
results = fob.compute_visualization(
    dataset,
    embeddings=embeddings,
    method="umap",
    brain_key="cbir-umap",
)

# Lança a aplicação
session = fo.launch_app(dataset)
session.show_brain("cbir-umap")
