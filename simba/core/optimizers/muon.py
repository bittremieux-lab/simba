"""Local single-device Muon optimizer implementation.

This is a lightweight adaptation of the public Muon optimizer design for
single-device training in SIMBA. Muon is applied only to selected matrix-shaped
parameters, while auxiliary Adam updates scalar/vector and excluded parameters.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def zeropower_via_newtonschulz5(grad: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate orthogonalization via Newton-Schulz iterations."""
    if grad.ndim == 4:
        grad = grad.view(len(grad), -1)
    if grad.ndim < 2:
        raise ValueError("Muon requires at least 2D parameters")

    a, b, c = (3.4445, -4.7750, 2.0315)
    x = grad.bfloat16()
    transposed = False
    if x.size(-2) > x.size(-1):
        x = x.mT
        transposed = True

    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        a_mat = x @ x.mT
        b_mat = b * a_mat + c * a_mat @ a_mat
        x = a * x + b_mat @ x

    if transposed:
        x = x.mT
    return x.to(grad.dtype)


@torch.no_grad()
def muon_update(
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    beta: float = 0.95,
    ns_steps: int = 5,
    nesterov: bool = True,
) -> torch.Tensor:
    """Compute Muon update from gradient and momentum state."""
    momentum_buffer.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum_buffer, beta) if nesterov else momentum_buffer
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    if update.ndim >= 2:
        update *= max(1.0, update.size(-2) / update.size(-1)) ** 0.5
    return update


@torch.no_grad()
def adam_update(
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: int,
    betas: tuple[float, float],
    eps: float,
) -> torch.Tensor:
    """Compute Adam-style parameter update."""
    exp_avg.lerp_(grad, 1 - betas[0])
    exp_avg_sq.lerp_(grad.square(), 1 - betas[1])
    exp_avg_corrected = exp_avg / (1 - betas[0] ** step)
    exp_avg_sq_corrected = exp_avg_sq / (1 - betas[1] ** step)
    return exp_avg_corrected / (exp_avg_sq_corrected.sqrt() + eps)


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """Single-device Muon + Adam hybrid optimizer.

    Parameter groups must include `use_muon=True` or `False`.
    """

    def __init__(self, param_groups: list[dict]):
        normalized_groups = []
        for group in param_groups:
            if "use_muon" not in group:
                raise ValueError("Each optimizer group must define 'use_muon'")

            new_group = dict(group)
            if new_group["use_muon"]:
                new_group["lr"] = new_group.get("lr", 0.02)
                new_group["momentum"] = new_group.get("momentum", 0.95)
                new_group["weight_decay"] = new_group.get("weight_decay", 0.0)
                new_group["ns_steps"] = new_group.get("ns_steps", 5)
                new_group["nesterov"] = new_group.get("nesterov", True)
            else:
                new_group["lr"] = new_group.get("lr", 3e-4)
                new_group["betas"] = new_group.get("betas", (0.9, 0.95))
                new_group["eps"] = new_group.get("eps", 1e-10)
                new_group["weight_decay"] = new_group.get("weight_decay", 0.0)
            normalized_groups.append(new_group)

        super().__init__(normalized_groups, defaults={})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for param in group["params"]:
                    if param.grad is None:
                        continue
                    state = self.state[param]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(param)
                    update = muon_update(
                        param.grad,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                        ns_steps=group["ns_steps"],
                        nesterov=group["nesterov"],
                    )
                    param.mul_(1 - group["lr"] * group["weight_decay"])
                    param.add_(update.reshape(param.shape), alpha=-group["lr"])
            else:
                for param in group["params"]:
                    if param.grad is None:
                        continue
                    state = self.state[param]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(param)
                        state["exp_avg_sq"] = torch.zeros_like(param)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        param.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )
                    param.mul_(1 - group["lr"] * group["weight_decay"])
                    param.add_(update, alpha=-group["lr"])
        return loss
