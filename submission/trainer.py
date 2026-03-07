import os
import math
import torch
from torch import nn
from typing import Tuple, List, Optional
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

# speed up convolutions for fixed-size inputs on GPU
torch.backends.cudnn.benchmark = True


class Trainer:
    """
    Training helper with:
      - fit(train_loader, val_loader, epochs)
      - per-epoch F1 (macro over 2 labels)
      - early stopping on best mean F1
      - checkpoint save/restore
      - ONNX export (dynamic batch)
      - optional ReduceLROnPlateau scheduler (attach at runtime via self.scheduler)
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        cuda: bool = False,
    ):
        self.model = model
        self.criterion = criterion
        self.device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = optimizer
        self.early_stopping_patience = 6
        self._best_f1 = -1.0
        self._best_state_f1 = None
        self.scheduler = None  # set externally if desired

    # ---------------------------
    # Core training API
    # ---------------------------
    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int = 50,
        verbose: bool = True,
    ) -> Tuple[List[float], List[float]]:
        assert self.optimizer is not None, "Trainer.optimizer must be set before calling fit()"

        os.makedirs(".", exist_ok=True)
        train_losses: List[float] = []
        val_losses: List[float] = []
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            # ---- Train ----
            self.model.train()
            running = 0.0
            n = 0
            for xb, yb in train_loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                loss = self.train_step(xb, yb)
                bs = xb.size(0)
                running += float(loss) * bs
                n += bs
            epoch_train = running / max(n, 1)
            train_losses.append(epoch_train)

            # ---- Validate ----
            self.model.eval()
            v_running = 0.0
            v_n = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device, non_blocking=True)
                    yb = yb.to(self.device, non_blocking=True)
                    v_loss = self.val_test_step(xb, yb)
                    bs = xb.size(0)
                    v_running += float(v_loss) * bs
                    v_n += bs
            epoch_val = v_running / max(v_n, 1)
            val_losses.append(epoch_val)

            # ---- Per-epoch F1 ----
            mean_f1, f1_per_label, _, _ = self.compute_f1(val_loader, threshold=0.5)

            if verbose:
                print(
                    f"Epoch {epoch:03d}: "
                    f"train_loss={epoch_train:.6f}  "
                    f"val_loss={epoch_val:.6f}  "
                    f"mean_F1={mean_f1:.3f}  "
                    f"F1_per_label={np.round(f1_per_label, 3)}"
                )

            # ---- Scheduler (optional) ----
            if self.scheduler is not None:
                self.scheduler.step(epoch_val)

            # ---- Early stopping on best F1 ----
            if mean_f1 > self._best_f1 + 1e-6:
                self._best_f1 = float(mean_f1)
                self._best_state_f1 = {k: v.cpu() for k, v in self.model.state_dict().items()}
                patience_counter = 0
                self.save_checkpoint("checkpoint_best.pt")
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    if verbose:
                        print(f"Early stopping on F1 (patience={self.early_stopping_patience}).")
                    break

        # Restore best-F1 weights for downstream export/eval
        if self._best_state_f1 is not None:
            self.model.load_state_dict(self._best_state_f1)

        return train_losses, val_losses

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        self.optimizer.zero_grad(set_to_none=True)
        pred = self.model(x)
        loss = self.criterion(pred, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def val_test_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        pred = self.model(x)
        loss = self.criterion(pred, y)
        return loss.item()

    @torch.no_grad()
    def compute_f1(self, val_loader, threshold: float = 0.5):
        """Macro-F1 over the two labels, computed on the validation loader."""
        self.model.eval()
        all_y, all_p = [], []
        for xb, yb in val_loader:
            xb = xb.to(self.device)
            yb = yb.cpu().numpy()
            preds = self.model(xb).cpu().numpy()
            preds_bin = (preds >= threshold).astype(int)
            all_y.append(yb)
            all_p.append(preds_bin)
        Y = np.vstack(all_y)
        P = np.vstack(all_p)
        prec, rec, f1, _ = precision_recall_fscore_support(Y, P, average=None, zero_division=0)
        return f1.mean(), f1, prec, rec

    # ---------------------------
    # Checkpoints / export
    # ---------------------------
    def save_checkpoint(self, path: str = "checkpoint.pt") -> None:
        torch.save(
            {"model_state": self.model.state_dict(), "best_f1": self._best_f1},
            path,
        )

    def restore_checkpoint(self, epoch: Optional[int] = None, path: str = "checkpoint.pt") -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self._best_f1 = ckpt.get("best_f1", -1.0)

    def save_onnx(self, filename: str = "model.onnx") -> None:
        """Export ONNX with dynamic batch dim (compatible with the unit tests)."""
        self.model.eval()
        dummy = torch.randn(1, 3, 300, 300, device=self.device)
        torch.onnx.export(
            self.model,
            dummy,
            filename,
            input_names=["input"],
            output_names=["output"],
            opset_version=12,
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
