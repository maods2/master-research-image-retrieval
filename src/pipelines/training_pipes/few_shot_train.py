from tqdm import tqdm
import torch
from torch.nn import functional as F
from pipelines.training_pipes.base_trainer import BaseTrainer
from utils.checkpoint_utils import save_model_and_log_artifact
from metrics.metric_base import MetricLoggerBase

from typing import Dict, Any, Callable, Tuple
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from utils.dataloader_utils import create_balanced_db_and_query
from utils.metrics_utils import compute_metrics


class FewShotTrain(BaseTrainer):
    def __init__(self, config: dict):
        self.config = config
        super().__init__()

    def prototypical_loss(
        self,
        support_embeddings,
        support_labels,
        query_embeddings,
        query_labels,
        n_way,
    ):
        prototypes = torch.stack(
            [
                support_embeddings[support_labels == i].mean(0)
                for i in range(n_way)
            ]
        )
        dists = torch.cdist(query_embeddings, prototypes)
        log_p = (-dists).log_softmax(dim=1)
        loss = F.nll_loss(log_p, query_labels)
        acc = (log_p.argmax(1) == query_labels).float().mean().item()
        return loss, acc

    def eval_few_shot_classification(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        support_set: Tuple[torch.Tensor, torch.Tensor],
        device: str,
        config: Dict[str, Any],
        logger: Callable,
    ) -> Dict[str, Any]:
        """
        Evaluate few-shot learning classification using support and query sets.

        Args:
            model: Model with feature embedding capability.
            test_loader: Dataloader providing (support, s_lbls, query, q_lbls) tuples.
            support_set: Tuple containing support embeddings and labels.
            config: Dictionary with configuration parameters. Expected: device.
            logger: Logging function.

        Returns:
            Dictionary with evaluation metrics.
        """

        all_preds = []
        all_labels = []
        model.eval()

        with torch.no_grad():
            for query, q_lbls in tqdm(test_loader, desc='Evaluating'):
                # Remove batch dim [1, N, ...] -> [N, ...]
                support = support_set[0].to(device)
                s_lbls = support_set[1].to(device)
                query = query.to(device)
                q_lbls = q_lbls.to(device)

                # Embed support and query
                emb_s = model(support)  # [n_support, D]
                emb_q = model(query)    # [n_query, D]

                # Compute class prototypes
                prototypes = model.compute_prototypes(
                    emb_s, s_lbls
                )  # [n_way, D]

                # Calculate euclidean distance between query and prototypes
                preds = model.predict_with_prototypes(emb_q, prototypes)

                all_preds.append(preds.cpu().numpy())
                all_labels.append(q_lbls.cpu().numpy())

        y_true = np.concatenate(all_labels)
        y_pred = np.concatenate(all_preds)

        return compute_metrics(y_true, y_pred, logger)

    def train_one_epoch(self, model, optimizer, dataloader, device, epoch):
        model.train()
        running_loss = 0.0
        running_acc = 0.0

        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}')

        for batch in progress_bar:
            support, s_lbls, query, q_lbls = batch
            support = support.squeeze(0).to(device)
            s_lbls = s_lbls.squeeze(0).to(device)
            query = query.squeeze(0).to(device)
            q_lbls = q_lbls.squeeze(0).to(device)

            optimizer.zero_grad()
            emb_s = model(support)
            emb_q = model(query)

            loss, acc = self.prototypical_loss(
                emb_s, s_lbls, emb_q, q_lbls, self.config['model']['n_way']
            )

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_acc += acc

            progress_bar.set_postfix(loss=loss.item(), acc=acc)

        avg_loss = running_loss / len(dataloader)
        avg_acc = running_acc / len(dataloader)

        return avg_loss, avg_acc, (support, s_lbls)

    def __call__(
        self,
        model,
        loss_fn,  # Não usado nesse caso, pois a loss está embutida
        optimizer,
        train_loader,
        test_loader,
        config,
        logger,
        metric_logger: MetricLoggerBase,
    ):
        device = config.get(
            'device', 'cuda' if torch.cuda.is_available() else 'cpu'
        )
        model.to(device)
        epochs = config['training']['epochs']
        patience = config['training'].get('early_stopping_patience', 10)

        min_loss = float('inf')
        epochs_without_improvement = 0
        checkpoint_path = None
        train_history = {
            'loss': [],
            'acc': [],
            'acc_val': [],
            'f1_score_val': [],
        }

        test_loader.dataset.k_shot = 1
        test_loader.dataset.validation_dataset = True

        for epoch in range(epochs):
            avg_loss, avg_acc, support_set = self.train_one_epoch(
                model, optimizer, train_loader, device, epoch
            )

            metrics = self.eval_few_shot_classification(
                model, test_loader, support_set, device, config, logger
            )

            logger.info(
                f'[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}'
            )
            print(
                f'[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}'
            )

            train_history['loss'].append(avg_loss)
            train_history['acc'].append(avg_acc)
            train_history['acc_val'].append(metrics['accuracy'])
            train_history['f1_score_val'].append(metrics['f1_score'])

            (
                should_stop,
                min_loss,
                epochs_without_improvement,
                checkpoint_path,
            ) = self.save_model_if_best(
                model=model,
                metric=avg_loss,
                best_metric=min_loss,
                epochs_without_improvement=epochs_without_improvement,
                checkpoint_path=checkpoint_path,
                config=config,
                metric_logger=metric_logger,
                mode='loss',
            )

            if should_stop:
                logger.info(
                    f'Early stopping triggered after {epochs_without_improvement} epochs with no improvement.'
                )
                print(
                    f'Early stopping triggered after {epochs_without_improvement} epochs with no improvement.'
                )
                break

        train_history['last_epoch_metrics'] = metrics
        metric_logger.log_json(train_history, 'train_metrics')

        return model
