from typing import Callable

import torch
from nnunetv2.utilities.ddp_allgather import AllGatherGrad
from torch import nn
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn


class TverskyLoss(nn.Module):
    """
    Tversky Loss implementation for handling class imbalance.

    Tversky index is a generalization of Dice coefficient:
    TI = TP / (TP + α*FP + β*FN)

    Where:
    - α controls the penalty for False Positives
    - β controls the penalty for False Negatives

    For small target structures:
    - Use α < 0.5 and β > 0.5 to penalize False Negatives more than False Positives
    - Example: α=0.3, β=0.7 emphasizes recall over precision

    When α = β = 0.5, Tversky Loss equals Dice Loss
    """

    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True,
                 smooth: float = 1., ddp: bool = True, alpha: float = 0.3, beta: float = 0.7):
        """
        Args:
            apply_nonlin: Activation function (softmax or sigmoid)
            batch_dice: Compute dice over entire batch or per sample
            do_bg: Include background class in loss computation
            smooth: Smoothing factor to avoid division by zero
            ddp: Use distributed data parallel
            alpha: Weight for False Positives (default: 0.3)
            beta: Weight for False Negatives (default: 0.7)
        """
        super(TverskyLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp
        self.alpha = alpha
        self.beta = beta

        # Validate alpha and beta
        assert alpha >= 0 and beta >= 0, "alpha and beta must be non-negative"
        assert alpha + beta > 0, "alpha + beta must be positive"

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # Get true positives, false positives, and false negatives
        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        if self.ddp and self.batch_dice:
            tp = AllGatherGrad.apply(tp).sum(0)
            fp = AllGatherGrad.apply(fp).sum(0)
            fn = AllGatherGrad.apply(fn).sum(0)

        # Tversky index calculation
        # TI = TP / (TP + α*FP + β*FN)
        nominator = tp
        denominator = tp + self.alpha * fp + self.beta * fn

        tversky = (nominator + self.smooth) / (torch.clip(denominator + self.smooth, 1e-8))

        if not self.do_bg:
            if self.batch_dice:
                tversky = tversky[1:]
            else:
                tversky = tversky[:, 1:]

        tversky = tversky.mean()

        # Return negative because we want to minimize loss (maximize Tversky index)
        return -tversky


class MemoryEfficientTverskyLoss(nn.Module):
    """
    Memory-efficient version of Tversky Loss.
    Saves GPU memory by computing statistics more efficiently.
    """

    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True,
                 smooth: float = 1., ddp: bool = True, alpha: float = 0.3, beta: float = 0.7):
        """
        Args:
            apply_nonlin: Activation function (softmax or sigmoid)
            batch_dice: Compute dice over entire batch or per sample
            do_bg: Include background class in loss computation
            smooth: Smoothing factor to avoid division by zero
            ddp: Use distributed data parallel
            alpha: Weight for False Positives (default: 0.3)
            beta: Weight for False Negatives (default: 0.7)
        """
        super(MemoryEfficientTverskyLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp
        self.alpha = alpha
        self.beta = beta

        # Validate alpha and beta
        assert alpha >= 0 and beta >= 0, "alpha and beta must be non-negative"
        assert alpha + beta > 0, "alpha + beta must be positive"

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # make everything shape (b, c)
        axes = tuple(range(2, x.ndim))

        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            if x.shape == y.shape:
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, y.long(), 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)

        # this one MUST be outside the with torch.no_grad(): context. Otherwise no gradients for you
        if not self.do_bg:
            x = x[:, 1:]

        if loss_mask is None:
            intersect = (x * y_onehot).sum(axes)  # True Positives
            sum_pred = x.sum(axes)
        else:
            intersect = (x * y_onehot * loss_mask).sum(axes)
            sum_pred = (x * loss_mask).sum(axes)

        # Calculate FP and FN
        # TP = intersect
        # FP = sum_pred - intersect
        # FN = sum_gt - intersect
        tp = intersect
        fp = sum_pred - intersect
        fn = sum_gt - intersect

        if self.batch_dice:
            if self.ddp:
                tp = AllGatherGrad.apply(tp).sum(0)
                fp = AllGatherGrad.apply(fp).sum(0)
                fn = AllGatherGrad.apply(fn).sum(0)

            tp = tp.sum(0)
            fp = fp.sum(0)
            fn = fn.sum(0)

        # Tversky index: TI = TP / (TP + α*FP + β*FN)
        tversky = (tp + self.smooth) / (torch.clip(tp + self.alpha * fp + self.beta * fn + self.smooth, 1e-8))

        tversky = tversky.mean()

        # Return negative to minimize loss (maximize Tversky index)
        return -tversky


if __name__ == '__main__':
    from nnunetv2.utilities.helpers import softmax_helper_dim1

    # Test case
    pred = torch.rand((2, 3, 32, 32, 32))
    ref = torch.randint(0, 3, (2, 32, 32, 32))

    # Test TverskyLoss
    tl = TverskyLoss(apply_nonlin=softmax_helper_dim1, batch_dice=True, do_bg=False,
                     smooth=1e-5, ddp=False, alpha=0.3, beta=0.7)
    tl_mem = MemoryEfficientTverskyLoss(apply_nonlin=softmax_helper_dim1, batch_dice=True,
                                        do_bg=False, smooth=1e-5, ddp=False, alpha=0.3, beta=0.7)

    res_tl = tl(pred, ref)
    res_tl_mem = tl_mem(pred, ref)

    print(f"TverskyLoss: {res_tl.item():.6f}")
    print(f"MemoryEfficientTverskyLoss: {res_tl_mem.item():.6f}")
    print(f"Difference: {abs(res_tl.item() - res_tl_mem.item()):.6f}")

    # Test that alpha=0.5, beta=0.5 gives similar results to Dice
    from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss

    tl_dice_equivalent = TverskyLoss(apply_nonlin=softmax_helper_dim1, batch_dice=True,
                                     do_bg=False, smooth=1e-5, ddp=False, alpha=0.5, beta=0.5)
    dl = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, batch_dice=True,
                      do_bg=False, smooth=1e-5, ddp=False)

    res_tl_dice = tl_dice_equivalent(pred, ref)
    res_dl = dl(pred, ref)

    print(f"\nTversky (α=0.5, β=0.5): {res_tl_dice.item():.6f}")
    print(f"Dice Loss: {res_dl.item():.6f}")
    print(f"Difference: {abs(res_tl_dice.item() - res_dl.item()):.6f}")
