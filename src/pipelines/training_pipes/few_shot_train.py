from tqdm import tqdm
import torch
from torch.nn import functional as F
from pipelines.training_pipes.base_trainer import BaseTrainer
from utils.checkpoint_utils import save_model_and_log_artifact
from metrics.metric_base import MetricLoggerBase


class FewShotTrain(BaseTrainer):
    def __init__(self, config: dict):
        self.config = config
        super().__init__()

    def prototypical_loss(self, support_embeddings, support_labels, query_embeddings, query_labels, n_way):
        prototypes = torch.stack([
            support_embeddings[support_labels == i].mean(0)
            for i in range(n_way)
        ])
        dists = torch.cdist(query_embeddings, prototypes)
        log_p = (-dists).log_softmax(dim=1)
        loss = F.nll_loss(log_p, query_labels)
        acc = (log_p.argmax(1) == query_labels).float().mean().item()
        return loss, acc

    def train_one_epoch(self, model, optimizer, dataloader, device, epoch):
        model.train()
        running_loss = 0.0
        running_acc = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

        for batch in progress_bar:
            support, s_lbls, query, q_lbls = batch
            support = support.squeeze(0).to(device)
            s_lbls = s_lbls.squeeze(0).to(device)
            query = query.squeeze(0).to(device)
            q_lbls = q_lbls.squeeze(0).to(device)

            optimizer.zero_grad()
            emb_s = model(support)
            emb_q = model(query)

            loss, acc = self.prototypical_loss(emb_s, s_lbls, emb_q, q_lbls, self.config['n_way'])

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_acc += acc

            progress_bar.set_postfix(loss=loss.item(), acc=acc)

        avg_loss = running_loss / len(dataloader)
        avg_acc = running_acc / len(dataloader)

        return avg_loss, avg_acc

    def __call__(
        self,
        model,
        loss_fn,  # Não usado nesse caso, pois a loss está embutida
        optimizer,
        train_loader,
        test_loader,
        config,
        logger,
        metric_logger: MetricLoggerBase
    ):
        device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        epochs = config['training']['epochs']

        min_loss = float("inf")
        checkpoint_path = None
        train_history = {"loss": [], "acc": []}

        for epoch in range(epochs):
            avg_loss, avg_acc = self.train_one_epoch(model, optimizer, train_loader, device, epoch)

            logger.info(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")
            print(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")

            train_history["loss"].append(avg_loss)
            train_history["acc"].append(avg_acc)

            if avg_loss < min_loss:
                min_loss = avg_loss
                checkpoint_path = save_model_and_log_artifact(metric_logger, config, model, filepath=checkpoint_path)

        metric_logger.log_json(train_history, "train_metrics")

        return model
