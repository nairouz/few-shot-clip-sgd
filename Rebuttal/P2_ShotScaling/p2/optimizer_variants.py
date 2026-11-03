from __future__ import annotations
import copy
import torch
from .update_budget import measured_update, snapshot_parameters

def build_optimizer(variant, parameters):
    params = list(parameters)
    if variant == "sgd":
        return torch.optim.SGD(params, lr=1e-4, momentum=0, weight_decay=0)
    if variant in {"adam_raw", "adam_m0"}:
        return torch.optim.Adam(params, lr=1e-4, betas=(.9, .999), eps=1e-8, weight_decay=0)
    raise ValueError(variant)

def proposed_budget(optimizer, parameters):
    parameters = list(parameters)
    before, state = snapshot_parameters(parameters), copy.deepcopy(optimizer.state_dict())
    optimizer.step()
    budget = measured_update(before, parameters)["relative_update_budget"]
    for parameter, value in zip(parameters, before):
        parameter.data.copy_(value)
    optimizer.load_state_dict(state)
    return budget

def configure_m0_first_step(optimizer, parameters, epsilon_budget=1e-30):
    parameters = list(parameters)
    raw = proposed_budget(optimizer, parameters)
    reference = proposed_budget(torch.optim.SGD(parameters, lr=1e-4, momentum=0, weight_decay=0), parameters)
    multiplier = reference / (raw + epsilon_budget)
    effective_lr = 1e-4 * multiplier
    for group in optimizer.param_groups:
        group["lr"] = effective_lr
    # Correct fp32 application-rounding using non-mutating trial steps. Adam's
    # proposed update is linear in LR, but storing the parameter delta is not.
    # This keeps state untouched while tightening the actual applied budget.
    best_lr, best_error = effective_lr, float("inf")
    for _ in range(12):
        trial = proposed_budget(optimizer, parameters)
        if trial == 0:
            break
        error = abs(trial - reference) / (reference + epsilon_budget)
        if error < best_error:
            best_lr, best_error = effective_lr, error
        effective_lr *= reference / trial
        for group in optimizer.param_groups:
            group["lr"] = effective_lr
    effective_lr = best_lr
    for group in optimizer.param_groups:
        group["lr"] = effective_lr
    return {"raw_adam_first_step_budget": raw, "reference_sgd_first_step_budget": reference,
            "matching_multiplier": multiplier, "persistent_effective_lr": effective_lr}
