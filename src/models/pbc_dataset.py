import os
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms



class PromptDataset(Dataset):
    def __init__(self, root, num_prompts=10, query_pos_ratio=0.5, transform=None, 
                 image_extensions={'.jpg', '.jpeg', '.png', '.bmp'}):
        self.root = root
        self.num_prompts = num_prompts
        self.query_pos_ratio = query_pos_ratio
        self.transform = transform or self.default_transform()
        self.image_extensions = image_extensions

        # Estruturas para mapeamento
        self.class_to_images = {}
        self.samples = []
        self._build_dataset_mappings()

    def _build_dataset_mappings(self):
        """Constroi os mapeamentos de classes e caminhos de imagens"""
        for dirpath, _, filenames in os.walk(self.root):
            for fname in filenames:
                if os.path.splitext(fname)[1].lower() in self.image_extensions:
                    rel_path = os.path.relpath(dirpath, self.root)
                    class_name = os.path.normpath(rel_path).split(os.sep)[0]
                    
                    img_path = os.path.join(dirpath, fname)
                    self.samples.append((img_path, class_name))
                    self.class_to_images.setdefault(class_name, []).append(img_path)

        self.classes = list(self.class_to_images.keys())
        print(f"Dataset carregado com {len(self.samples)} amostras e {len(self.classes)} classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        query_path, query_label = self.samples[idx]
        query = self._load_image(query_path)
        
        is_positive = random.random() < self.query_pos_ratio
        
        # Lógica de amostragem corrigida
        if is_positive:
            # Positivos: mesma classe
            pos_imgs, pos_labels = self._sample_images_with_labels(
                target_class=query_label, 
                is_negative=False,
                exclude_path=query_path
            )
            # Negativos: todas outras classes
            neg_imgs, neg_labels = self._sample_images_with_labels(
                target_class=query_label,
                is_negative=True,
                exclude_path=None
            )
        else:
            # Positivos: classe aleatória diferente
            other_classes = [c for c in self.classes if c != query_label]
            random_class = random.choice(other_classes) if other_classes else query_label
            
            pos_imgs, pos_labels = self._sample_images_with_labels(
                target_class=random_class,
                is_negative=False,
                exclude_path=None
            )
            # Negativos: mesma classe (mas exclui query)
            neg_imgs, neg_labels = self._sample_images_with_labels(
                target_class=query_label,
                is_negative=False,
                exclude_path=query_path
            )

        return {
            'query': query,
            'query_label': torch.tensor(1 if is_positive else 0, dtype=torch.float32),
            'class_name': query_label,
            'pos_imgs': torch.stack(pos_imgs),
            'pos_imgs_labels': pos_labels,
            'neg_imgs': torch.stack(neg_imgs),
            'neg_imgs_labels': neg_labels
        }

    def _get_sampling_classes(self, query_label, is_positive):
        """Determina classes para amostragem baseado no tipo da query"""
        if is_positive:
            return query_label, None  # Positivos: mesma classe, Negativos: outras
        else:
            other_classes = [c for c in self.classes if c != query_label]
            return random.choice(other_classes) if other_classes else query_label, query_label

    def _sample_images_with_labels(self, target_class, is_negative=False, exclude_path=None):
        """Amostra imagens e retorna com seus labels reais
        Args:
            target_class: Classe alvo para positivos (ou classe a evitar para negativos se is_negative=True)
            is_negative: Se True, amostra de todas as classes EXCETO target_class
            exclude_path: Caminho a ser excluído da amostragem
        """
        if is_negative:
            # Amostra de todas as classes exceto a target_class
            candidates = []
            for cls, paths in self.class_to_images.items():
                if cls != target_class:
                    candidates.extend(paths)
        else:
            # Amostra apenas da target_class
            candidates = self.class_to_images.get(target_class, []).copy()

        # Remove o caminho a ser excluído se existir
        if exclude_path and exclude_path in candidates:
            candidates.remove(exclude_path)

        # Seleciona imagens ou usa padding
        if len(candidates) >= self.num_prompts:
            selected_paths = random.sample(candidates, self.num_prompts)
        else:
            selected_paths = candidates.copy()

        # Carrega imagens e obtém seus labels reais
        images = []
        labels = []
        for path in selected_paths:
            images.append(self._load_image(path))
            # Obtém o label real do caminho da imagem
            labels.append(self._get_class_from_path(path))

        # Completa com padding se necessário
        while len(images) < self.num_prompts:
            images.append(torch.zeros(3, 224, 224))
            labels.append("padding")

        return images[:self.num_prompts], labels[:self.num_prompts]

    def _get_class_from_path(self, path):
        """Obtém o nome da classe a partir do caminho da imagem"""
        rel_path = os.path.relpath(os.path.dirname(path), self.root)
        return os.path.normpath(rel_path).split(os.sep)[0]

    def _load_image(self, path):
        """Carrega imagem ou retorna tensor zerado em caso de erro"""
        try:
            img = Image.open(path).convert("RGB")
            return self.transform(img)
        except:
            return torch.zeros(3, 224, 224)

    def default_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# Função para collate personalizado
def custom_collate(batch):
    """Agrupa batch mantendo a estrutura complexa dos dados"""
    return {
        'query': torch.stack([item['query'] for item in batch]),
        'query_label': torch.stack([item['query_label'] for item in batch]),
        # 'query_is_positive': torch.tensor([item['query_is_positive'] for item in batch]),
        'pos_imgs': torch.stack([item['pos_imgs'] for item in batch]),
        'pos_imgs_labels': [item['pos_imgs_labels'] for item in batch],
        'neg_imgs': torch.stack([item['neg_imgs'] for item in batch]),
        'neg_imgs_labels': [item['neg_imgs_labels'] for item in batch],
        'class_name': [item['class_name'] for item in batch]
    }

# Exemplo de uso
if __name__ == '__main__':
    dataset = PromptDataset("datasets/final/terumo/train", num_prompts=10)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=custom_collate)
    
    for batch in dataloader:
        print("\nBatch completo:")
        print(f"Query shape: {batch['query'].shape}")
        print(f"Query labels: {batch['query_label']}")
        print(f"Query is positive: {batch['query_is_positive']}")
        
        print("\nPrimeiro exemplo do batch:")
        print(f"Positives shape: {batch['pos_imgs'][0].shape}")
        print(f"Positives labels: {batch['pos_imgs_labels'][0]}")
        print(f"Negatives shape: {batch['neg_imgs'][0].shape}")
        print(f"Negatives labels: {batch['neg_imgs_labels'][0]}")
        break