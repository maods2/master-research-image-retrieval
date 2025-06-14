import torch
import torch.nn.functional as F
from torch import nn


class NTXentLoss(nn.Module):
    """
    https://arxiv.org/pdf/2205.03169
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        # z1, z2: (N, D)
        N, _ = z1.shape
        z = torch.cat([z1, z2], dim=0)  # (2N, D)
        z = F.normalize(z, dim=1)
        sim_matrix = torch.matmul(z, z.T) / self.temperature  # (2N,2N)
        mask = torch.eye(2 * N, device=sim_matrix.device).bool()
        sim_matrix = sim_matrix.masked_fill(mask, -9e15)

        # positives are diagonal offsets
        positives = torch.cat(
            [sim_matrix[i, i + N].unsqueeze(0) for i in range(N)]
            + [sim_matrix[i + N, i].unsqueeze(0) for i in range(N)],
            dim=0,
        )
        exp_sim = torch.exp(sim_matrix)
        denom = exp_sim.sum(dim=1)
        loss = -torch.log(torch.exp(positives) / denom)
        return loss.mean()


class SupConLoss(torch.nn.Module):
    """
    https://arxiv.org/pdf/2004.11362
    """

    def __init__(self, config):
        super().__init__()
        self.temperature = config.get('temperature', 0.07)

    def forward(self, features, labels):
        device = features.device
        batch_size = labels.shape[0]
        labels = labels.contiguous().view(-1, 1)

        mask = torch.eq(labels, labels.T).float().to(device)
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        logits = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )

        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )

        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        loss = -mean_log_prob_pos
        loss = loss.mean()

        return loss


class ProxyNCALoss(torch.nn.Module):
    """
    https://arxiv.org/pdf/1703.07464
    """

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
    """
    https://arxiv.org/pdf/1904.06627
    """

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
    """
    https://arxiv.org/pdf/1801.07698
    """

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
    """
    https://papers.nips.cc/paper_files/paper/2016/file/6b180037abbebea991d8b1232f8a8ca9-Paper.pdf
    """

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
