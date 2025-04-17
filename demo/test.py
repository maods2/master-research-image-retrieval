import fiftyone as fo
import fiftyone.utils.data as foud
import os
import fiftyone as fo
import numpy as np
import pathlib
from pathlib import Path
import numpy as np

# 1) Monkey‑patch: todo PosixPath vira WindowsPath
pathlib.PosixPath = pathlib.WindowsPath

data = np.load("./artifacts/glomerulo/embeddings_uni/embeddings_2025-04-16_04-16-09.npz", allow_pickle=True)
db_embeddings = data["db_embeddings"]
db_labels_bin = data["db_labels"]
db_paths = data["db_path"]
query_embeddings = data["query_embeddings"]
class_mapping = data['class_mapping'].item()
db_labels = np.array([class_mapping[i] for i in db_labels_bin])
dataset = fo.Dataset()
samples = []

root = r"D:\dev\master-research-image-retrieval"




for idx, (path, label, emb) in enumerate(zip(db_paths, db_labels, db_embeddings)):

    full_path = os.path.join(root, path)
    print(full_path)
# for filepath, label in zip(filepaths, labels):
    sample = fo.Sample(filepath=full_path)
    sample["ground_truth"] = fo.Classification(label=label)
    samples.append(sample)

dataset.add_samples(samples)

session = fo.launch_app(dataset, port=5151)
# print("App lançado em http://localhost:5151")
session.wait()

# import fiftyone as fo
# import fiftyone.zoo as foz

# dataset = foz.load_zoo_dataset("quickstart")
# session = fo.launch_app(dataset)
# session.wait()