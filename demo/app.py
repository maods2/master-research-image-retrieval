import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import streamlit as st
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import os
from src.models.model_factory import get_model
from src.dataloaders.transform_factory import get_transforms
import torch
# model_config = {
#     'name': 'model_name',
#     'model_name': "vit_large_patch16_224",
#     'load_checkpoint': True,
#     'checkpoint_path': None,
#     'resize': [224, 224],
#     'normalize': True,
#     'to_tensor': True
# }
# transform = get_transforms(model_config)

# Função para carregar dados de embedding
@st.cache_data
def load_embeddings_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    class_mapping = data['class_mapping'].item()
    db_embeddings = data["db_embeddings"]
    db_paths = data["db_path"]
    db_labels = np.array([class_mapping[i] for i in data["db_labels"]])
    return db_embeddings, db_paths, db_labels

# Função para gerar embedding da imagem de consulta
def load_model():
    model_name = "uni"
    model_config = {
        'name': model_name,
        'model_name': "vit_large_patch16_224",
        'num_classes': 6,
        'load_checkpoint': False,
        'checkpoint_path': None,
        'resize': [224, 224],
        'normalize': {'mean': (0.485, 0.456, 0.406), 'std': [0.229, 0.224, 0.225]},
        'to_tensor': True
    }
    transform = get_transforms(model_config)
    model = get_model(model_config)
    return model, transform

def call_model(img, model, transform):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    model.eval()
    with torch.no_grad():
        # Preprocessamento da imagem
        img = np.array(img)
        img = transform(image=img)['image'].to(device)
        
        # Geração do embedding
        embedding = model(img.unsqueeze(0))
        embedding = embedding.squeeze(0).cpu().numpy()

    return embedding.reshape(1, -1)


# Função para buscar os top-k mais similares
def get_top_k(query_embedding, db_embeddings, db_paths, db_labels, k=5):
    similarities = cosine_similarity(query_embedding, db_embeddings).flatten()
    top_k_indices = np.argsort(-similarities)[:k]
    return [(db_paths[i], db_labels[i], similarities[i]) for i in top_k_indices]

# Configurações da UI
st.title('Image Retrieval System com Embeddings Reais')
st.sidebar.header('Configurações')

# Lista de bancos de dados disponíveis (caminho para os arquivos .npz)
available_dbs = {
    'Glomerulo': "artifacts/glomerulo/embeddings_uni/embeddings_2025-04-16_04-16-09.npz",
    'BRACS': "artifacts/bracs-resized/embeddings_uni/embeddings_2025-04-16_03-05-03.npz",
    'CRC-VAL-HE-7K': "artifacts/CRC-VAL-HE-7K-splitted/embeddings_uni/embeddings_2025-04-16_03-33-52.npz",
    'Ovarian Cancer': "artifacts/ovarian-cancer-splitted/embeddings_uni/embeddings_2025-04-16_04-35-27.npz",
    'Skin Cancer': "artifacts/skin-cancer-splitted/embeddings_uni/embeddings_2025-04-16_04-58-53.npz",
}

selected_db_name = st.sidebar.selectbox("Selecione o banco de dados", list(available_dbs.keys()))
top_k = st.sidebar.slider("Número de imagens similares (K)", min_value=1, max_value=20, value=5)

# Carregamento do banco de dados
db_path = available_dbs[selected_db_name]
db_embeddings, db_image_paths, db_labels = load_embeddings_npz(db_path)
model, transform = load_model()

# Interface de upload
st.header("Imagem de Consulta")
uploaded_file = st.file_uploader("Envie uma imagem", type=["png", "jpg", "jpeg"])

if uploaded_file:
    query_image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(query_image, caption="Imagem de Consulta", use_container_width=True)

    if st.button("Buscar similares"):
        query_embedding = call_model(query_image, model, transform)  # substitua por modelo real
        results = get_top_k(query_embedding, db_embeddings, db_image_paths, db_labels, top_k)

        with col2:
            st.header("Top-K Imagens Similares")
            for path, label, similarity in results:
                if os.path.exists(path):
                    st.image(path, caption=f"{label} — Similaridade: {similarity:.4f}", use_container_width=True)
                else:
                    st.warning(f"Imagem não encontrada: {path}")
