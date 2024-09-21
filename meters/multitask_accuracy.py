import torch
from torchmetrics import Metric
from typing import Any


class MultitaskAccuracy(Metric):
    def __init__(self, num_labels: int = 2, top_k: int = 1, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.top_k = top_k
        self.nlabels = num_labels

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: tuple[torch.Tensor, ...], target: tuple[torch.Tensor, ...]):
        bs = target[0].shape[0]
        all_correct = torch.zeros(self.top_k, bs).type(torch.ByteTensor).to(self.device)

        for output, label in zip(preds, target):
            _, max_k_idx = output.topk(self.top_k, dim=1, largest=True, sorted=True)
            # Flip batch_size, class_count as .view doesn't work on non-contiguous
            max_k_idx = max_k_idx.t()
            correct_for_task = max_k_idx.eq(label.view(1, -1).expand_as(max_k_idx))
            all_correct.add_(correct_for_task)

        self.correct += torch.ge(all_correct.float().sum(0), self.nlabels).sum()
        self.total += all_correct.shape[1]

    def compute(self):
        return self.correct.float() / self.total
