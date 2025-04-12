import torch
from torch.utils.data import Subset
from collections import defaultdict
import random



def create_balanced_subset(dataset, total_samples, seed=42):
    """
    Create a balanced subset of the dataset with total_samples, 
    trying to maintain class proportions.
    """
    label_to_indices = defaultdict(list)

    # Group indices by class
    for idx, (_, label) in enumerate(dataset):
        label_to_indices[label].append(idx)

    num_classes = len(label_to_indices)
    samples_per_class = total_samples // num_classes

    random.seed(seed)
    selected_indices = []

    for label, indices in label_to_indices.items():
        if len(indices) < samples_per_class:
            raise ValueError(f"Not enough samples for class {label} (has {len(indices)}, needs {samples_per_class})")
        selected = random.sample(indices, samples_per_class)
        selected_indices.extend(selected)

    # Optional: shuffle the final indices
    random.shuffle(selected_indices)

    return Subset(dataset, selected_indices)


def create_balanced_db_and_query(dataset, total_db_samples, total_query_samples, seed=42):
    """
    Creates two balanced subsets (database and query) from the given dataset, each with a specified number of total samples.
    The subsets preserve class balance as much as possible.
    """
    label_to_indices = defaultdict(list)

    # Group indices by class
    for idx, label in enumerate(dataset.labels):
        label_to_indices[label].append(idx)

    num_classes = len(label_to_indices)
    db_per_class = total_db_samples // num_classes
    query_per_class = total_query_samples // num_classes

    random.seed(seed)
    db_indices = []
    query_indices = []

    for label, indices in label_to_indices.items():
        if len(indices) < db_per_class + query_per_class:
            raise ValueError(f"Not enough samples for class {label} (has {len(indices)}, needs {db_per_class + query_per_class})")

        # Shuffle indices of the current class
        random.shuffle(indices)
        
        # Split for DB and Query
        query_split = indices[:query_per_class]
        db_split = indices[query_per_class:query_per_class + db_per_class]

        query_indices.extend(query_split)
        db_indices.extend(db_split)

    random.shuffle(query_indices)
    random.shuffle(db_indices)

    return Subset(dataset, db_indices), Subset(dataset, query_indices)


if __name__ == "__main__":
    # Example dataset
    from torchvision import datasets, transforms

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.FakeData(transform=transform, size=600, num_classes=5)

    db_subset, query_subset = create_balanced_db_and_query(
        dataset=dataset,
        total_db_samples=400,
        total_query_samples=60,
        seed=42
        )
    
    balanced_subset_q = create_balanced_subset(dataset, total_samples=50)

    # Check class distribution
    from collections import Counter
    labels = [label for _, label in db_subset]
    print("Balanced class counts:", Counter(labels))
    labels = [label for _, label in query_subset]
    print("Balanced class counts:", Counter(labels))
    labels = [label for _, label in balanced_subset_q]
    print("Balanced class counts:", Counter(labels))