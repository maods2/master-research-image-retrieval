import torch
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        # z1, z2: (N, D)
        N, _ = z1.shape
        z = torch.cat([z1, z2], dim=0)  # (2N, D)
        z = F.normalize(z, dim=1)
        sim_matrix = torch.matmul(z, z.T) / self.temperature  # (2N,2N)
        mask = torch.eye(2*N, device=sim_matrix.device).bool()
        sim_matrix = sim_matrix.masked_fill(mask, -9e15)

        # positives are diagonal offsets
        positives = torch.cat([sim_matrix[i, i+N].unsqueeze(0) for i in range(N)] +
                              [sim_matrix[i+N, i].unsqueeze(0) for i in range(N)], dim=0)
        exp_sim = torch.exp(sim_matrix)
        denom = exp_sim.sum(dim=1)
        loss = -torch.log(torch.exp(positives) / denom)
        return loss.mean()
    
class SupervisedContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Create mask for positive pairs (same class)
        labels = labels.unsqueeze(0)
        mask = torch.eq(labels, labels.T).float()

        # Exclude diagonal (self-similarity)
        logits_mask = torch.ones_like(mask) - torch.eye(
            embeddings.size(0), device=mask.device
        )
        masked_sim = sim_matrix * logits_mask

        # Compute log probabilities
        exp_sim = torch.exp(masked_sim)
        log_prob = masked_sim - torch.log(exp_sim.sum(dim=1, keepdim=True))

        # Average log-likelihood over positive pairs
        loss = -(mask * log_prob).sum(dim=1) / mask.sum(dim=1)
        return loss.mean()


class ProxyNCALoss(torch.nn.Module):
    def __init__(self, num_classes, embedding_dim, temperature=0.1):
        super().__init__()
        self.proxies = torch.nn.Parameter(
            torch.randn(num_classes, embedding_dim)
        )
        self.temperature = temperature

    def forward(self, embeddings, labels):
        # Normalize embeddings and proxies
        embeddings = F.normalize(embeddings, p=2, dim=1)
        proxies = F.normalize(self.proxies, p=2, dim=1)

        # Compute distances to proxies
        dist = torch.cdist(embeddings, proxies) / self.temperature

        # Select distances for ground-truth classes
        loss = -torch.log(
            torch.exp(-dist[torch.arange(len(labels)), labels]).sum()
            / torch.exp(-dist).sum()
        )
        return loss


class MultiSimilarityLoss(torch.nn.Module):
    def __init__(self, alpha=2.0, beta=50.0, base=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.base = base

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        sim_matrix = torch.matmul(embeddings, embeddings.T)

        # Positive and negative masks
        pos_mask = torch.eq(
            labels.unsqueeze(0), labels.unsqueeze(1)
        ).float() - torch.eye(len(labels), device=embeddings.device)
        neg_mask = torch.ne(labels.unsqueeze(0), labels.unsqueeze(1)).float()

        # Loss components
        pos_loss = (1.0 / self.alpha) * torch.log(
            1
            + torch.sum(
                torch.exp(-self.alpha * (sim_matrix - self.base)) * pos_mask,
                dim=1,
            )
        )
        neg_loss = (1.0 / self.beta) * torch.log(
            1
            + torch.sum(
                torch.exp(self.beta * (sim_matrix - self.base)) * neg_mask,
                dim=1,
            )
        )

        return (pos_loss + neg_loss).mean()


class ArcFaceLoss(torch.nn.Module):
    def __init__(self, num_classes, embedding_dim, margin=0.5, scale=64.0):
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn(embedding_dim, num_classes))
        self.margin = margin
        self.scale = scale

    def forward(self, embeddings, labels):
        # Normalize embeddings and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        W = F.normalize(self.W, p=2, dim=0)

        # Compute logits
        logits = torch.matmul(embeddings, W) * self.scale

        # Add angular margin
        one_hot = F.one_hot(labels, num_classes=W.size(1))
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        marginal_logits = torch.cos(theta + self.margin * one_hot)

        # Final loss
        loss = F.cross_entropy(marginal_logits, labels)
        return loss


class NPairLoss(torch.nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Mask for positive pairs (diagonal for N-pair)
        pos_mask = torch.eye(len(labels), device=embeddings.device)
        neg_mask = 1 - pos_mask

        # Loss: Push positives apart from all negatives in the batch
        loss = -torch.log(
            torch.exp(sim_matrix * pos_mask).sum(dim=1)
            / torch.exp(sim_matrix * neg_mask).sum(dim=1)
        )
        return loss.mean()


