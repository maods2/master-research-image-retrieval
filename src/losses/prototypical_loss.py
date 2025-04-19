import torch
import torch.nn.functional as F


class PrototypicalLoss:
    def __init__(self, config):
        self.n_way = config['n_way']

    def __call__(
        self,
        support_embeddings,
        support_labels,
        query_embeddings,
        query_labels,
    ):
        # Calcular protótipos
        prototypes = torch.stack(
            [
                support_embeddings[support_labels == i].mean(0)
                for i in range(self.n_way)
            ]
        )

        # Distância euclidiana
        dists = torch.cdist(query_embeddings, prototypes)

        # Probabilidades via softmax negativo da distância
        log_p_y = (-dists).log_softmax(dim=1)

        # Loss e acurácia
        loss = F.nll_loss(log_p_y, query_labels)
        acc = (log_p_y.argmax(dim=1) == query_labels).float().mean().item()

        return loss, acc
