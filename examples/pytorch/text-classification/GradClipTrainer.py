from transformers import Trainer, AdamW
from transformers.trainer_pt_utils import get_parameter_names
from transformers.trainer_utils import ShardedDDPOption
from transformers.integrations import is_fairscale_available
from transformers.dependency_versions_check import dep_version_check
from torch import nn
from typing import Callable, Iterable, Optional, Tuple, Union
from transformers.utils.versions import require_version
import torch
import math
if is_fairscale_available():
    dep_version_check("fairscale")
    import fairscale
    from fairscale.nn.data_parallel import FullyShardedDataParallel as FullyShardedDDP
    from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
    from fairscale.nn.wrap import auto_wrap
    from fairscale.optim import OSS
    from fairscale.optim.grad_scaler import ShardedGradScaler

class ClipValueAdamW(AdamW):
    def __init__(
            self,
            params: Iterable[nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
            max_clip_value: float = 0.15
    ):
        require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0[")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0[")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, **defaults)
        # record all numbers of clpped terms
        self.clip_number_history = []
        self.max_clip_val = max_clip_value
    pass

    def clip(self, p, update, g_i, p_i, name):
        self.clip_number_history.append((name,tuple(p.shape), (p, g_i, p_i),
                                         (update > self.max_clip_val).sum().item(),
                                         torch.numel(p.data)
                                         ))
        update = torch.clamp(update, max=self.max_clip_val)
        return update

    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (:obj:`Callable`, `optional`): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for g_i, group in enumerate(self.param_groups):
            for p_i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                grad = p.grad.data
                # pre-correction clipping
                grad = self.clip(p, grad, g_i, p_i, "pre-correction-update")
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                # used for post-update, but according to recent meeting, we should do clipping before error correction terms are computed1yy
                # dummy_data = torch.zeros_like(p.data)
                # update = torch.addcdiv(dummy_data, exp_avg, denom, value=-step_size)
                # update = self.clip(p, update, g_i, p_i, name="normal update")
                # p.data.add_(update)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=(-group["lr"] * group["weight_decay"]))
                    # used for post-update, but according to recent meeting, we should do clipping before error correction terms are computed1yy
                    # update = p.data * -group["lr"] * group["weight_decay"]
                    # update = self.clip(p, update, g_i, p_i, name="weight decay update")
                    # p.data.add_(update)

        return loss


# we need to sublcass Trainer, because we need some way to inform the max_clip_value
class GradValueClipTrainer(Trainer):
    def create_optimizer(self):
        # by default, use AdamW to conform with Devlin et al., 2019
        # here the main change is that we change clip_norm_max behavior of AdamW, so it can be used to implement clip_value
        decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]
        # if not self.args.adafactor:
        optimizer_cls = ClipValueAdamW
        optimizer_kwargs = {
            "betas": (self.args.adam_beta1, self.args.adam_beta2),
            "eps": self.args.adam_epsilon,
            "max_clip_value": self.args.max_clip_value
        }
        optimizer_kwargs["lr"] = self.args.learning_rate
        print(optimizer_kwargs)
        if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            self.optimizer = OSS(
                params=optimizer_grouped_parameters,
                optim=optimizer_cls,
                **optimizer_kwargs,
            )
        else:
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        pass
