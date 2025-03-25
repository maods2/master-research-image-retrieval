# import torch
# import timm

# # Criar um modelo ViT pré-treinado
# model = timm.create_model("vit_base_patch16_224", pretrained=True)

# # Colocar o modelo em modo de avaliação
# model.eval()

# # Criar uma função para capturar TODOS os embeddings
# def extract_all_embeddings(model, x):
#     with torch.no_grad():
#         x = model.patch_embed(x)  # Criar patches e embeddings
#         x = torch.cat([model.cls_token.expand(x.shape[0], -1, -1), x], dim=1)  # Adicionar CLS Token
#         x = model.pos_drop(x + model.pos_embed)  # Adicionar embeddings de posição e dropout
#         for blk in model.blocks:  # Passar pelos blocos Transformer
#             x = blk(x)
#         embeddings = x  # Retorna todos os tokens (CLS + patches)
#     return embeddings

# # Criar uma entrada fictícia (imagem de 224x224 com 3 canais)
# x = torch.randn(1, 3, 224, 224)  # Um batch com 1 imagem

# # Extrair todos os embeddings
# embeddings = extract_all_embeddings(model, x)
# print(embeddings.shape)  # Deve ser algo como (1, 197, 768)

# patch_embeddings = embeddings[:, 1:, :]  # Remove CLS Token
# print(patch_embeddings.shape)  # Deve ser (1, 196, 768)

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# -------------------------------
# 1. ViT Backbone Compartilhado
# -------------------------------
class ViTBackbone(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True):
        super(ViTBackbone, self).__init__()
        # Cria o modelo ViT sem a camada de classificação
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
    
    def forward(self, x):
        # Processa x e retorna o token CLS (assumindo que o modelo retorna [B, tokens, embed_dim])
        out = self.model(x)
        if out.dim() == 3:
            return out[:, 0, :]  # retorna o token CLS
        return out

# -------------------------------
# 2. Prompt Encoder
# -------------------------------
class PromptEncoder(nn.Module):
    def __init__(self, embed_dim, num_transformer_layers=2, num_heads=4):
        super(PromptEncoder, self).__init__()
        # Projeção para os embeddings dos prompts
        self.proj = nn.Linear(embed_dim, embed_dim)
        # Camadas Transformer para processar a concatenação dos embeddings
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        # Ajuste para concatenar embeddings positivos e negativos
        self.adjust = nn.Linear(2 * embed_dim, embed_dim)
    
    def forward(self, pos_embeds, neg_embeds):
        """
        pos_embeds, neg_embeds: tensores de shape [B, embed_dim]
        """
        # 1. Projeção dos embeddings
        pos_proj = self.proj(pos_embeds)  # [B, embed_dim]
        neg_proj = self.proj(neg_embeds)  # [B, embed_dim]
        
        # 2. (Opcional) Pode calcular a similaridade cosseno, mascaramento, etc.
        # Exemplo: cos_sim = F.cosine_similarity(pos_proj.unsqueeze(1), neg_proj.unsqueeze(0), dim=-1)
        
        # 3. Concatenar e ajustar dimensões
        combined = torch.cat([pos_proj, neg_proj], dim=-1)  # [B, 2*embed_dim]
        combined = self.adjust(combined).unsqueeze(1)        # [B, 1, embed_dim]
        
        # 4. Processamento com Transformer
        transformer_out = self.transformer(combined)         # [B, 1, embed_dim]
        
        # 5. Agregação (removendo dimensão de sequência)
        prompt_context = transformer_out.squeeze(1)          # [B, embed_dim]
        return prompt_context

# -------------------------------
# 3. Attention Fusion
# -------------------------------
class AttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super(AttentionFusion, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    
    def forward(self, query_embed, prompt_context):
        """
        query_embed: [B, embed_dim]
        prompt_context: [B, embed_dim]
        """
        # Expandir para incluir a dimensão de sequência (tamanho 1)
        query = prompt_context.unsqueeze(1)      # [B, 1, embed_dim]
        key   = prompt_context.unsqueeze(1)     # [B, 1, embed_dim]
        value = query_embed.unsqueeze(1)     # [B, 1, embed_dim]
        attn_output, _ = self.multihead_attn(query, key, value)
        fused = attn_output.squeeze(1)          # [B, embed_dim]
        return fused

# -------------------------------
# 4. Classifier Head
# -------------------------------
class ClassifierHead(nn.Module):
    def __init__(self, embed_dim):
        super(ClassifierHead, self).__init__()
        self.fc = nn.Linear(embed_dim, 1)  # Saída binária
        
    def forward(self, x):
        logits = self.fc(x)
        return logits

# -------------------------------
# 5. Modelo Completo: Promptable Binary Classifier
# -------------------------------
class PromptableBinaryClassifier(nn.Module):
    def __init__(self, vit_model_name='vit_base_patch16_224', embed_dim=768):
        super(PromptableBinaryClassifier, self).__init__()
        # Utiliza um único backbone ViT compartilhado para todas as entradas
        self.shared_backbone = ViTBackbone(vit_model_name)
        
        # Prompt Encoder para processar os embeddings dos prompts
        self.prompt_encoder = PromptEncoder(embed_dim)
        
        # Módulo de Attention Fusion para combinar query e o contexto dos prompts
        self.attn_fusion = AttentionFusion(embed_dim)
        
        # Classifier head para a saída final binária
        self.classifier = ClassifierHead(embed_dim)
    
    def forward(self, query_img, pos_prompts, neg_prompts):
        """
        query_img: tensor [B, 3, 224, 224]
        pos_prompts: tensor [B, N, 3, 224, 224]
        neg_prompts: tensor [B, N, 3, 224, 224]
        """
        B = query_img.size(0)
        
        # Processamento da imagem de consulta
        q_embed = self.shared_backbone(query_img)  # [B, embed_dim]
        
        # Processamento dos prompts positivos
        B, N, C, H, W = pos_prompts.shape
        pos_prompts = pos_prompts.view(B * N, C, H, W)
        pos_embeds = self.shared_backbone(pos_prompts)  # [B*N, embed_dim]
        pos_embeds = pos_embeds.view(B, N, -1)           # [B, N, embed_dim]
        pos_agg = pos_embeds.mean(dim=1)                 # Agrega via média -> [B, embed_dim]
        
        # Processamento dos prompts negativos
        B, N, C, H, W = neg_prompts.shape
        neg_prompts = neg_prompts.view(B * N, C, H, W)
        neg_embeds = self.shared_backbone(neg_prompts)  # [B*N, embed_dim]
        neg_embeds = neg_embeds.view(B, N, -1)           # [B, N, embed_dim]
        neg_agg = neg_embeds.mean(dim=1)                 # [B, embed_dim]
        
        # Obter o contexto dos prompts via Prompt Encoder
        prompt_context = self.prompt_encoder(pos_agg, neg_agg)  # [B, embed_dim]
        
        # Fusão dos embeddings da query com o contexto dos prompts
        fused = self.attn_fusion(q_embed, prompt_context)  # [B, embed_dim]
        
        # Classificação final
        logits = self.classifier(fused)  # [B, 1]
        return logits

# -------------------------------
# Exemplo de uso
# -------------------------------
if __name__ == '__main__':
    # Instancia o modelo
    model = PromptableBinaryClassifier()
    
    # Exemplo de batch com 2 imagens de consulta
    query_img = torch.randn(2, 3, 224, 224)
    
    # Exemplo com 3 prompts positivos e 3 negativos por amostra
    pos_prompts = torch.randn(2, 10, 3, 224, 224)
    neg_prompts = torch.randn(2, 10, 3, 224, 224)
    
    # Forward
    logits = model(query_img, pos_prompts, neg_prompts)
    print("Logits:", logits.shape)
    print("Logits:", logits)
