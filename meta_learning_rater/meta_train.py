"""
MAML meta-learning loop for the TSRater.

Inner loop: signSGD over all support batches for inner_steps epochs.
Outer loop: per-task meta_optimizer zero_grad / backward / step.

Evaluation:
  Few-shot fine-tune adapted model on a small support slice of test,
  then measure pairwise ranking accuracy on the held-out test set.
"""
from __future__ import annotations

import copy
import random

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from meta_learning_rater.model import ScoreModel, BradleyTerryLoss


class MetaLearner:
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        meta_lr: float = 2.5e-4,
        inner_lr: float = 4e-3,
        inner_steps: int = 14,
        device: str | None = None,
    ):
        self.device      = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.inner_lr    = inner_lr
        self.inner_steps = inner_steps
        self.loss_fn     = BradleyTerryLoss()

        self.meta_model     = ScoreModel(input_dim, hidden_dim, num_layers).to(self.device)
        self.meta_optimizer = optim.SGD(self.meta_model.parameters(), lr=meta_lr)

    # ── Inner loop (signSGD, matches meta_rater/maml.py) ─────────────────────

    def _inner_adapt(self, support_loader: DataLoader) -> ScoreModel:
        """
        Inner loop: inner_steps epochs over the full support set using signSGD.
        create_graph=True retained for compatibility with original code.
        """
        fmodel = copy.deepcopy(self.meta_model)
        fmodel.train()

        for _ in range(self.inner_steps):
            for emb_a, emb_b, p in support_loader:
                emb_a = emb_a.to(self.device)
                emb_b = emb_b.to(self.device)
                p     = p.to(self.device)

                scores_a = fmodel(emb_a)
                scores_b = fmodel(emb_b)
                loss = self.loss_fn(scores_a, scores_b, p)

                grads = torch.autograd.grad(loss, fmodel.parameters(), create_graph=True)
                with torch.no_grad():
                    for param, grad in zip(fmodel.parameters(), grads):
                        param -= self.inner_lr * grad.sign()

        return fmodel

    # ── Outer loop ────────────────────────────────────────────────────────────

    def meta_train(
        self,
        task_datasets: list[dict],
        meta_batch_size: int = 10,
        data_batch_size: int = 16,
        epochs: int = 7,
    ) -> None:
        for epoch in range(epochs):
            sampled = random.sample(
                task_datasets, min(meta_batch_size, len(task_datasets))
            )
            total_meta_loss = 0.0

            for task in sampled:
                support_loader = DataLoader(
                    task["support"], batch_size=data_batch_size,
                    shuffle=True, num_workers=0,
                )
                query_loader = DataLoader(
                    task["query"], batch_size=data_batch_size,
                    shuffle=True, num_workers=0,
                )

                fmodel = self._inner_adapt(support_loader)

                # Compute meta loss on query set
                fmodel.eval()
                meta_loss = 0.0
                for emb_a, emb_b, p in query_loader:
                    emb_a = emb_a.to(self.device)
                    emb_b = emb_b.to(self.device)
                    p     = p.to(self.device)
                    meta_loss += self.loss_fn(fmodel(emb_a), fmodel(emb_b), p)

                self.meta_optimizer.zero_grad()
                meta_loss.backward()
                self.meta_optimizer.step()
                total_meta_loss += meta_loss.item()

            print(f"[Epoch {epoch+1}/{epochs}]  meta_loss={total_meta_loss / len(sampled):.4f}")

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        task_datasets: list[dict],
        data_batch_size: int = 16,
        few_shot_ratio: float = 0.05,
        adaptation_steps: int = 10,
        adaptation_lr: float = 5e-3,
    ) -> float:
        accs = []
        for task in task_datasets:
            acc = self._eval_one_task(
                task["test"],
                data_batch_size=data_batch_size,
                few_shot_ratio=few_shot_ratio,
                adaptation_steps=adaptation_steps,
                adaptation_lr=adaptation_lr,
            )
            accs.append(acc)
            name = task.get("name", "?")
            print(f"  {name:<30}  acc={acc:.4f}")

        mean_acc = sum(accs) / len(accs) if accs else 0.0
        print(f"\n  Mean accuracy ({len(accs)} tasks): {mean_acc:.4f}")
        return mean_acc

    def _eval_one_task(
        self,
        test_dataset,
        data_batch_size: int,
        few_shot_ratio: float,
        adaptation_steps: int,
        adaptation_lr: float,
    ) -> float:
        total = len(test_dataset)
        few_shot_len = max(1, int(total * few_shot_ratio))
        eval_len = total - few_shot_len
        if eval_len < 1:
            return 0.0

        support_set, eval_set = random_split(test_dataset, [few_shot_len, eval_len])

        model = copy.deepcopy(self.meta_model).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=adaptation_lr)
        model.train()
        support_loader = DataLoader(support_set, batch_size=1, shuffle=True)
        for _ in range(adaptation_steps):
            for emb_a, emb_b, p in support_loader:
                emb_a = emb_a.to(self.device)
                emb_b = emb_b.to(self.device)
                p     = p.to(self.device)
                loss = self.loss_fn(model(emb_a), model(emb_b), p)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        correct = total_samples = 0
        eval_loader = DataLoader(eval_set, batch_size=data_batch_size, shuffle=False)
        with torch.no_grad():
            for emb_a, emb_b, p in eval_loader:
                emb_a = emb_a.to(self.device)
                emb_b = emb_b.to(self.device)
                p     = p.to(self.device)
                pred  = (model(emb_b) > model(emb_a)).float()
                correct       += (pred == (p > 0.5).float()).sum().item()
                total_samples += len(p)

        return correct / total_samples if total_samples else 0.0