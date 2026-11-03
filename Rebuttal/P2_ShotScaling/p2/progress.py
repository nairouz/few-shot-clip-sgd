from __future__ import annotations
import math
Q_VALUES = (0.0, 0.10, 0.33, 0.67, 1.0)

def targets(total_updates: int):
    if total_updates < 1:
        raise ValueError("total_updates must be positive")
    result = []
    for q in Q_VALUES:
        step = 0 if q == 0 else min(total_updates, max(1, math.floor(q * total_updates + .5)))
        result.append({"requested_q": q, "target_step": step, "actual_progress": step / total_updates})
    if len({x["target_step"] for x in result}) != len(result):
        raise ValueError(f"non-unique diagnostic targets for T={total_updates}")
    return result
