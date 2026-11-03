from __future__ import annotations

def snapshot_parameters(parameters):
    return [p.detach().clone() for p in parameters]

def measured_update(before, parameters):
    pre = sum(float(x.float().square().sum()) for x in before) ** .5
    delta = sum(float((p.detach().float() - x.float()).square().sum()) for x, p in zip(before, parameters)) ** .5
    return {"pre_step_parameter_norm": pre, "update_norm": delta,
            "relative_update_budget": delta / pre if pre else float("nan")}

def gradient_norm(parameters):
    return sum(float(p.grad.detach().float().square().sum())
               for p in parameters if p.grad is not None) ** .5
