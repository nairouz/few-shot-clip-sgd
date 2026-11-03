# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

"""Numerically faithful optimizer instrumentation for the P5 SigLIP controls.

The functions in this module deliberately measure parameter changes around
native ``torch.optim`` calls.  In particular, no Adam update is reconstructed
from ``exp_avg``/``exp_avg_sq``.  This matters for foreach/fused implementations,
epsilon placement, bias correction, and future PyTorch changes.

The public API is intentionally independent of the CLIP training loop:

* :func:`build_siglip_scope_map` validates the global, tower, and Transformer
  block partition.
* :func:`native_optimizer_step` records an optimizer's actual parameter delta.
* :func:`adam_m0_first_step` performs the one-time first-step LR match.
* :func:`adam_mt_step` retains Adam's native nominal-LR moment update while
  realizing global, encoderwise, or blockwise scales through a bounded
  two-shadow search of native Adam effective-LR steps before storage rounding.
* :class:`WarmupSchedule` implements the exact P1 warmup schedules.
* :func:`synchronized_step_decision` gates paired trajectories on non-finite
  gradients and AMP skip indications.
* :func:`compute_parameter_drift` measures fp32 drift from initialization.

All norm reductions explicitly convert operands to fp32 before accumulation.
Finite matching scales are never clipped.
"""

from __future__ import annotations

import copy
import contextlib
import math
import re
from dataclasses import dataclass, replace
from typing import (
    Callable,
    Dict,
    Generic,
    Iterable,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.optim import Adam, Optimizer, SGD


TensorMap = Dict[str, torch.Tensor]
NamedParameterSource = Union[
    nn.Module,
    Iterable[Tuple[str, nn.Parameter]],
    "ScopeMap",
]

GLOBAL_SCOPE = "global"
ENCODER_SCOPES = ("encoder/vision", "encoder/text")

_BLOCK_PATTERNS = (
    ("vision", re.compile(r"^(?:module\.)?vision_model\.encoder\.layers\.(\d+)\.")),
    ("text", re.compile(r"^(?:module\.)?text_model\.encoder\.layers\.(\d+)\.")),
)


class InstrumentationError(RuntimeError):
    """Base error for a P1 instrumentation invariant violation."""


class ScopeMappingError(InstrumentationError):
    """A trainable parameter does not form a complete, unique CLIP partition."""


class NonFiniteMatchError(InstrumentationError):
    """A matching budget or scale is non-finite."""


class PrecisionMatchError(InstrumentationError):
    """No fp32-representable scalar update met the matching tolerance."""


@dataclass(frozen=True)
class ScopeEntry:
    """The unique encoder/block assignment of one trainable parameter."""

    name: str
    parameter: nn.Parameter
    encoder: str
    block: int

    @property
    def encoder_scope(self) -> str:
        return f"encoder/{self.encoder}"

    @property
    def block_scope(self) -> str:
        return f"block/{self.encoder}/{self.block}"


class ScopeMap:
    """A validated partition of trainable CLIP parameters.

    Every entry belongs to the global scope, exactly one encoder, and exactly
    one Transformer block.  Use :func:`build_clip_scope_map` instead of
    constructing this class directly.
    """

    def __init__(self, entries: Sequence[ScopeEntry], blocks_per_encoder: int):
        if not entries:
            raise ScopeMappingError("No trainable parameters were provided.")
        self.entries: Tuple[ScopeEntry, ...] = tuple(entries)
        self.blocks_per_encoder = int(blocks_per_encoder)
        self._by_name = {entry.name: entry for entry in self.entries}
        if len(self._by_name) != len(self.entries):
            raise ScopeMappingError("Trainable parameter names are not unique.")

    @property
    def named_parameters(self) -> Tuple[Tuple[str, nn.Parameter], ...]:
        return tuple((entry.name, entry.parameter) for entry in self.entries)

    @property
    def parameter_names(self) -> Tuple[str, ...]:
        return tuple(entry.name for entry in self.entries)

    @property
    def parameters(self) -> Tuple[nn.Parameter, ...]:
        return tuple(entry.parameter for entry in self.entries)

    def scope_for(self, parameter_name: str, level: str) -> str:
        """Return the canonical scope key for ``parameter_name``."""

        try:
            entry = self._by_name[parameter_name]
        except KeyError as exc:
            raise ScopeMappingError(
                f"Parameter {parameter_name!r} is absent from the scope map."
            ) from exc
        if level == "global":
            return GLOBAL_SCOPE
        if level == "encoder":
            return entry.encoder_scope
        if level == "block":
            return entry.block_scope
        raise ValueError(
            f"Unknown scope level {level!r}; expected 'global', 'encoder', or 'block'."
        )

    def groups(self, level: str, *, include_empty: bool = False) -> Dict[str, Tuple[str, ...]]:
        """Return ordered parameter-name groups for a matching level."""

        if level == "global":
            return {GLOBAL_SCOPE: self.parameter_names}
        if level == "encoder":
            keys = ENCODER_SCOPES
        elif level == "block":
            keys = tuple(
                f"block/{encoder}/{block}"
                for encoder in ("vision", "text")
                for block in range(self.blocks_per_encoder)
            )
        else:
            raise ValueError(
                f"Unknown scope level {level!r}; expected 'global', 'encoder', or 'block'."
            )

        grouped: MutableMapping[str, list[str]] = {key: [] for key in keys}
        for entry in self.entries:
            key = entry.encoder_scope if level == "encoder" else entry.block_scope
            grouped[key].append(entry.name)
        return {
            key: tuple(names)
            for key, names in grouped.items()
            if include_empty or names
        }

    def all_groups(self, *, include_empty: bool = False) -> Dict[str, Tuple[str, ...]]:
        """Return the global, two encoder, and block groups in one mapping."""

        result = self.groups("global", include_empty=include_empty)
        result.update(self.groups("encoder", include_empty=include_empty))
        result.update(self.groups("block", include_empty=include_empty))
        return result


def _named_parameters(
    source: NamedParameterSource,
    *,
    only_trainable: bool = True,
) -> Tuple[Tuple[str, nn.Parameter], ...]:
    if isinstance(source, ScopeMap):
        items = source.named_parameters
    elif isinstance(source, nn.Module):
        items = tuple(source.named_parameters())
    else:
        items = tuple(source)

    selected = tuple(
        (str(name), parameter)
        for name, parameter in items
        if not only_trainable or parameter.requires_grad
    )
    names = [name for name, _ in selected]
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ScopeMappingError(f"Duplicate parameter names: {duplicates}")

    identities: Dict[int, str] = {}
    aliases = []
    for name, parameter in selected:
        previous = identities.setdefault(id(parameter), name)
        if previous != name:
            aliases.append((previous, name))
    if aliases:
        raise ScopeMappingError(
            "A trainable parameter was supplied more than once (multiply mapped): "
            + ", ".join(f"{left!r}/{right!r}" for left, right in aliases)
        )
    return selected


def build_siglip_scope_map(
    source: NamedParameterSource,
    *,
    blocks_per_encoder: int = 12,
    require_all_blocks: bool = True,
) -> ScopeMap:
    """Build and validate the SigLIP global/tower/block scope partition.

    The accepted Hugging Face SigLIP names are
    ``vision_model.encoder.layers.<i>.*`` and
    ``text_model.encoder.layers.<i>.*`` (optionally prefixed by ``module.``).
    Any unmapped trainable parameter, duplicate parameter identity, invalid
    block index, or missing expected block raises :class:`ScopeMappingError`.

    ``blocks_per_encoder=12`` and ``require_all_blocks=True`` enforce the
    canonical ViT-B/16 map of 24 Transformer blocks.  Tests or reduced smoke
    models can explicitly request a smaller complete map.
    """

    if blocks_per_encoder <= 0:
        raise ValueError("blocks_per_encoder must be positive.")

    entries = []
    unmapped = []
    multiply_mapped = []
    invalid_blocks = []
    for name, parameter in _named_parameters(source, only_trainable=True):
        matches = []
        for encoder, pattern in _BLOCK_PATTERNS:
            match = pattern.match(name)
            if match is not None:
                matches.append((encoder, int(match.group(1))))
        if not matches:
            unmapped.append(name)
            continue
        if len(matches) != 1:
            multiply_mapped.append(name)
            continue
        encoder, block = matches[0]
        if not 0 <= block < blocks_per_encoder:
            invalid_blocks.append((name, block))
            continue
        entries.append(ScopeEntry(name, parameter, encoder, block))

    errors = []
    if unmapped:
        errors.append("unmapped=" + ", ".join(repr(name) for name in unmapped))
    if multiply_mapped:
        errors.append(
            "multiply_mapped=" + ", ".join(repr(name) for name in multiply_mapped)
        )
    if invalid_blocks:
        errors.append(
            "out_of_range="
            + ", ".join(f"{name!r}:block={block}" for name, block in invalid_blocks)
        )
    if errors:
        raise ScopeMappingError(
            "Every trainable parameter must map to exactly one encoder and block; "
            + "; ".join(errors)
        )

    result = ScopeMap(entries, blocks_per_encoder)
    if require_all_blocks:
        populated = set(result.groups("block").keys())
        expected = {
            f"block/{encoder}/{block}"
            for encoder in ("vision", "text")
            for block in range(blocks_per_encoder)
        }
        missing = sorted(expected - populated)
        if missing:
            raise ScopeMappingError(
                "The canonical block scope map is incomplete; missing scopes: "
                + ", ".join(missing)
            )
    return result


# Compatibility alias for inherited model-neutral optimizer helpers and tests.
build_clip_scope_map = build_siglip_scope_map


def snapshot_parameters(source: NamedParameterSource) -> TensorMap:
    """Clone all trainable parameters without changing dtype or device."""

    return {
        name: parameter.detach().clone()
        for name, parameter in _named_parameters(source, only_trainable=True)
    }


@torch.no_grad()
def restore_parameters(source: NamedParameterSource, snapshot: Mapping[str, torch.Tensor]) -> None:
    """Restore a snapshot after first validating its exact parameter-name set."""

    named = dict(_named_parameters(source, only_trainable=True))
    _assert_same_names(named, snapshot, "parameters", "snapshot")
    for name, parameter in named.items():
        parameter.copy_(snapshot[name])


def parameter_deltas(
    before: Mapping[str, torch.Tensor],
    source: NamedParameterSource,
) -> TensorMap:
    """Return actual ``after - before`` parameter changes."""

    named = dict(_named_parameters(source, only_trainable=True))
    _assert_same_names(named, before, "parameters", "before snapshot")
    return {
        name: parameter.detach().clone() - before[name]
        for name, parameter in named.items()
    }


def _assert_same_names(
    left: Mapping[str, object],
    right: Mapping[str, object],
    left_label: str,
    right_label: str,
) -> None:
    left_names, right_names = set(left), set(right)
    if left_names != right_names:
        missing = sorted(left_names - right_names)
        extra = sorted(right_names - left_names)
        raise InstrumentationError(
            f"{left_label}/{right_label} names differ: "
            f"missing_in_{right_label.replace(' ', '_')}={missing}, "
            f"extra_in_{right_label.replace(' ', '_')}={extra}"
        )


def fp32_l2_norm(tensors: Iterable[torch.Tensor]) -> float:
    """Compute an L2 norm with fp32 operands and fp64 accumulation."""

    total: Optional[torch.Tensor] = None
    for tensor in tensors:
        value = tensor.detach()
        if value.is_sparse:
            value = value.coalesce().values()
        value = value.to(dtype=torch.float32)
        square_sum = torch.sum(value * value, dtype=torch.float64)
        if total is None:
            total = torch.zeros((), dtype=torch.float64, device=square_sum.device)
        if square_sum.device != total.device:
            raise InstrumentationError("A single norm scope spans multiple devices.")
        total.add_(square_sum)
    if total is None:
        return 0.0
    return float(torch.sqrt(total).item())


@dataclass(frozen=True)
class ScopeBudget:
    """Absolute and relative update norms for one scope."""

    parameter_l2: float
    update_l2: float
    relative_budget: float


def _scope_budget(
    before: Mapping[str, torch.Tensor],
    deltas: Mapping[str, torch.Tensor],
    names: Sequence[str],
) -> ScopeBudget:
    parameter_l2 = fp32_l2_norm(before[name] for name in names)
    update_l2 = fp32_l2_norm(deltas[name] for name in names)
    if parameter_l2 > 0.0:
        relative = update_l2 / parameter_l2
    elif update_l2 == 0.0:
        relative = 0.0
    else:
        relative = math.inf
    return ScopeBudget(parameter_l2, update_l2, relative)


def compute_scope_budgets(
    before: Mapping[str, torch.Tensor],
    deltas: Mapping[str, torch.Tensor],
    scope_map: ScopeMap,
    *,
    levels: Sequence[str] = ("global", "encoder", "block"),
    include_empty: bool = False,
) -> Dict[str, ScopeBudget]:
    """Compute fp32 budgets for requested canonical scope levels."""

    expected = {name: None for name in scope_map.parameter_names}
    _assert_same_names(expected, before, "scope map", "before snapshot")
    _assert_same_names(expected, deltas, "scope map", "deltas")
    groups: Dict[str, Tuple[str, ...]] = {}
    for level in levels:
        groups.update(scope_map.groups(level, include_empty=include_empty))
    return {
        scope: _scope_budget(before, deltas, names)
        for scope, names in groups.items()
    }


@dataclass(frozen=True)
class MatchDecision:
    """Applied scale, analytic scale, and numerical-edge flags for one scope.

    ``matching_scale`` computes the analytic budget ratio and leaves
    ``analytic_scale`` as ``None``.  Storage-aware calibration changes
    ``scale`` to the scalar that realizes the requested budget after parameter
    storage rounding and retains the formula value in ``analytic_scale``.
    """

    scale: float
    both_zero: bool = False
    unmatchable: bool = False
    analytic_scale: Optional[float] = None
    storage_compensation: float = 1.0
    calibration_iterations: int = 0
    precalibration_relative_error: float = math.nan
    postcalibration_relative_error: float = math.nan
    matching_acceptance_tolerance: float = 1e-5
    storage_rounding_fallback: bool = False
    strict_calibration_exhausted: bool = False
    calibration_outcome: str = "strict"

    @property
    def formula_scale(self) -> float:
        """Return the pre-calibration analytic budget ratio."""

        return self.scale if self.analytic_scale is None else self.analytic_scale


def matching_scale(
    reference_budget: float,
    adam_budget: float,
    *,
    epsilon_budget: float = 1e-12,
    scope: str = GLOBAL_SCOPE,
) -> MatchDecision:
    """Return the P1 scale without clipping any finite value.

    Edge semantics are exact:

    * both budgets ``<= epsilon_budget``: scale 1, ``both_zero=True``;
    * Adam ``<= epsilon_budget`` but reference is larger: scale 1,
      ``unmatchable=True`` (the zero Adam delta remains zero);
    * otherwise use ``reference / (adam + epsilon_budget)``;
    * any non-finite input or output raises :class:`NonFiniteMatchError`.
    """

    if epsilon_budget < 0.0 or not math.isfinite(epsilon_budget):
        raise ValueError("epsilon_budget must be finite and non-negative.")
    if (
        not math.isfinite(reference_budget)
        or not math.isfinite(adam_budget)
        or reference_budget < 0.0
        or adam_budget < 0.0
    ):
        raise NonFiniteMatchError(
            f"Non-finite or negative budget in scope {scope!r}: "
            f"reference={reference_budget!r}, adam={adam_budget!r}"
        )
    if adam_budget <= epsilon_budget:
        if reference_budget <= epsilon_budget:
            return MatchDecision(scale=1.0, both_zero=True)
        return MatchDecision(scale=1.0, unmatchable=True)

    scale = reference_budget / (adam_budget + epsilon_budget)
    if not math.isfinite(scale):
        raise NonFiniteMatchError(
            f"Non-finite matching scale in scope {scope!r}: "
            f"reference={reference_budget!r}, adam={adam_budget!r}, "
            f"epsilon_budget={epsilon_budget!r}, scale={scale!r}"
        )
    return MatchDecision(scale=scale)


def compute_matching_scales(
    reference_budgets: Mapping[str, ScopeBudget],
    adam_budgets: Mapping[str, ScopeBudget],
    *,
    epsilon_budget: float = 1e-12,
) -> Dict[str, MatchDecision]:
    """Compute one unclipped decision for every matched scope."""

    _assert_same_names(
        reference_budgets, adam_budgets, "reference scopes", "Adam scopes"
    )
    return {
        scope: matching_scale(
            reference_budgets[scope].relative_budget,
            adam_budgets[scope].relative_budget,
            epsilon_budget=epsilon_budget,
            scope=scope,
        )
        for scope in reference_budgets
    }


CalibrationPayload = TypeVar("CalibrationPayload")


@dataclass
class _CalibrationResult(Generic[CalibrationPayload]):
    decisions: Dict[str, MatchDecision]
    budgets: Dict[str, ScopeBudget]
    payload: CalibrationPayload


@dataclass
class _CalibrationBracket:
    target: float
    low_scale: float = 0.0
    low_budget: float = 0.0
    high_scale: Optional[float] = None
    high_budget: Optional[float] = None
    ratio_attempted: bool = False
    iterations: int = 0
    best_error: float = math.inf
    best_scale: float = 0.0
    best_budget: float = 0.0
    exhausted: bool = False


def _realized_matching_error(
    actual_budget: float,
    target_budget: float,
    decision: MatchDecision,
    *,
    epsilon_budget: float,
) -> float:
    if decision.both_zero:
        return 0.0
    if decision.unmatchable:
        return math.nan
    if target_budget > epsilon_budget:
        return abs(actual_budget - target_budget) / target_budget
    return 0.0 if actual_budget <= epsilon_budget else math.inf


def _calibrated_decision(
    analytic: MatchDecision,
    *,
    scale: float,
    iterations: int,
    pre_error: float,
    post_error: float,
    acceptance_tolerance: float,
    storage_rounding_fallback: bool,
    strict_calibration_exhausted: bool,
    calibration_outcome: str,
) -> MatchDecision:
    formula_scale = analytic.formula_scale
    if formula_scale == 0.0:
        compensation = 1.0 if scale == 0.0 else math.inf
    else:
        compensation = scale / formula_scale
    if not math.isfinite(compensation):
        raise NonFiniteMatchError(
            "Non-finite storage compensation for analytic scale "
            f"{formula_scale!r} and applied scale {scale!r}."
        )
    return replace(
        analytic,
        scale=scale,
        analytic_scale=formula_scale,
        storage_compensation=compensation,
        calibration_iterations=iterations,
        precalibration_relative_error=pre_error,
        postcalibration_relative_error=post_error,
        matching_acceptance_tolerance=acceptance_tolerance,
        storage_rounding_fallback=storage_rounding_fallback,
        strict_calibration_exhausted=strict_calibration_exhausted,
        calibration_outcome=calibration_outcome,
    )


def _next_calibration_scale(
    bracket: _CalibrationBracket,
) -> Optional[float]:
    """Choose a deterministic safeguarded scalar for a monotone fp32 budget."""

    if bracket.high_scale is None:
        if bracket.low_scale <= 0.0:
            raise PrecisionMatchError(
                "Cannot expand a calibration bracket from a zero scale."
            )
        # A ratio iteration can converge only asymptotically on a quantized
        # staircase and never establish an upper bracket.  Doubling guarantees
        # progress; once an upper point exists, the safeguarded search below
        # rapidly contracts around the attainable fp32 plateau.
        candidate = bracket.low_scale * 2.0
        if not math.isfinite(candidate):
            raise NonFiniteMatchError(
                "Storage-aware calibration produced a non-finite upper scale."
            )
        return candidate

    assert bracket.high_budget is not None
    if not bracket.ratio_attempted and bracket.high_budget > 0.0:
        bracket.ratio_attempted = True
        candidate = (
            bracket.high_scale * bracket.target / bracket.high_budget
        )
        if bracket.low_scale < candidate < bracket.high_scale:
            return candidate

    candidate = 0.5 * (bracket.low_scale + bracket.high_scale)
    if candidate == bracket.low_scale or candidate == bracket.high_scale:
        bracket.exhausted = True
        return None
    return candidate


def _storage_aware_calibration(
    analytic_decisions: Mapping[str, MatchDecision],
    target_budgets: Mapping[str, ScopeBudget],
    evaluator: Callable[
        [Mapping[str, float]],
        Tuple[Mapping[str, ScopeBudget], CalibrationPayload],
    ],
    *,
    epsilon_budget: float,
    matching_tolerance: float,
    storage_rounding_tolerance: float,
    max_calibration_iterations: int,
) -> _CalibrationResult[CalibrationPayload]:
    """Calibrate scalar scales against actual parameter-storage changes.

    For a fixed raw update and non-negative scalar, each rounded parameter
    change is monotone in magnitude.  The scope L2 budget is therefore
    monotone, though it is a staircase rather than a linear function in fp32.
    This routine evaluates the exact final application path and performs a
    safeguarded scalar search only for scopes whose analytic ratio misses the
    strict realized-budget tolerance.  If the scalar fp32 lattice is exhausted,
    the best scalar may use the explicitly documented mixed-precision fallback
    tolerance.  It never changes the raw direction or individual coordinates.
    """

    if (
        not math.isfinite(matching_tolerance)
        or matching_tolerance < 0.0
    ):
        raise ValueError("matching_tolerance must be finite and non-negative.")
    if (
        not math.isfinite(storage_rounding_tolerance)
        or storage_rounding_tolerance < matching_tolerance
    ):
        raise ValueError(
            "storage_rounding_tolerance must be finite and no smaller than "
            "matching_tolerance."
        )
    if max_calibration_iterations <= 0:
        raise ValueError("max_calibration_iterations must be positive.")
    _assert_same_names(
        analytic_decisions,
        target_budgets,
        "matching decisions",
        "target budgets",
    )

    formula_scales = {
        scope: decision.formula_scale
        for scope, decision in analytic_decisions.items()
    }
    for scope, scale in formula_scales.items():
        if not math.isfinite(scale) or scale < 0.0:
            raise NonFiniteMatchError(
                f"Invalid analytic matching scale {scale!r} for scope {scope!r}."
            )

    realized_budgets, payload = evaluator(formula_scales)
    _assert_same_names(
        analytic_decisions,
        realized_budgets,
        "matching decisions",
        "realized budgets",
    )
    pre_errors: Dict[str, float] = {}
    brackets: Dict[str, _CalibrationBracket] = {}
    active = set()
    chosen_scales = dict(formula_scales)

    for scope, analytic in analytic_decisions.items():
        target = target_budgets[scope].relative_budget
        realized = realized_budgets[scope].relative_budget
        error = _realized_matching_error(
            realized,
            target,
            analytic,
            epsilon_budget=epsilon_budget,
        )
        pre_errors[scope] = error
        bracket = _CalibrationBracket(
            target=target,
            best_error=error if math.isfinite(error) else math.inf,
            best_scale=formula_scales[scope],
            best_budget=realized,
        )
        if realized < target:
            bracket.low_scale = formula_scales[scope]
            bracket.low_budget = realized
        else:
            bracket.high_scale = formula_scales[scope]
            bracket.high_budget = realized
        brackets[scope] = bracket
        if (
            not analytic.both_zero
            and not analytic.unmatchable
            and (not math.isfinite(error) or error > matching_tolerance)
        ):
            active.add(scope)

    for _ in range(max_calibration_iterations):
        if not active:
            break
        candidates = {}
        for scope in tuple(active):
            candidate = _next_calibration_scale(brackets[scope])
            if candidate is None:
                active.remove(scope)
                continue
            candidates[scope] = candidate
            chosen_scales[scope] = candidate
        if not candidates:
            break

        realized_budgets, payload = evaluator(chosen_scales)
        for scope in tuple(candidates):
            bracket = brackets[scope]
            bracket.iterations += 1
            realized = realized_budgets[scope].relative_budget
            analytic = analytic_decisions[scope]
            error = _realized_matching_error(
                realized,
                bracket.target,
                analytic,
                epsilon_budget=epsilon_budget,
            )
            if math.isfinite(error) and error < bracket.best_error:
                bracket.best_error = error
                bracket.best_scale = chosen_scales[scope]
                bracket.best_budget = realized
            if math.isfinite(error) and error <= matching_tolerance:
                active.remove(scope)
                continue
            if realized < bracket.target:
                if chosen_scales[scope] >= bracket.low_scale:
                    bracket.low_scale = chosen_scales[scope]
                    bracket.low_budget = realized
            else:
                if (
                    bracket.high_scale is None
                    or chosen_scales[scope] <= bracket.high_scale
                ):
                    bracket.high_scale = chosen_scales[scope]
                    bracket.high_budget = realized
    # Retain each unresolved scope's best observed scalar, then jointly
    # re-evaluate those independent scope choices through the exact storage
    # path.  Reaching the caller's iteration cap is *not* proof that the scalar
    # lattice was exhausted.  ``_next_calibration_scale`` marks exhaustion only
    # when the low/high floating-point endpoints are adjacent (their midpoint
    # equals an endpoint), so the relaxed fallback below remains evidence-based.
    unresolved = {
        scope
        for scope, bracket in brackets.items()
        if (
            not analytic_decisions[scope].both_zero
            and not analytic_decisions[scope].unmatchable
            and bracket.best_error > matching_tolerance
        )
    }
    for scope in unresolved:
        chosen_scales[scope] = brackets[scope].best_scale
    if unresolved:
        realized_budgets, payload = evaluator(chosen_scales)

    post_errors = {
        scope: _realized_matching_error(
            realized_budgets[scope].relative_budget,
            target_budgets[scope].relative_budget,
            analytic_decisions[scope],
            epsilon_budget=epsilon_budget,
        )
        for scope in analytic_decisions
    }
    failures = []
    fallback_scopes = set()
    for scope in sorted(unresolved):
        bracket = brackets[scope]
        error = post_errors[scope]
        if (
            bracket.exhausted
            and math.isfinite(error)
            and matching_tolerance < error <= storage_rounding_tolerance
        ):
            fallback_scopes.add(scope)
            continue
        search_status = (
            "adjacent_scalar_endpoints"
            if bracket.exhausted
            else "iteration_cap_without_exhaustion"
        )
        failures.append(
            f"{scope}: best_error={bracket.best_error:.9g}, "
            f"reapplied_error={error:.9g}, "
            f"best_scale={bracket.best_scale:.17g}, "
            f"best_budget={bracket.best_budget:.17g}, "
            f"target={bracket.target:.17g}, "
            f"strict_search_exhausted={int(bracket.exhausted)}, "
            f"search_status={search_status}"
        )
    if failures:
        raise PrecisionMatchError(
            "No scalar fp32 parameter update met the strict matching tolerance "
            f"{matching_tolerance:.9g}, and the documented storage-rounding "
            f"fallback {storage_rounding_tolerance:.9g} was unavailable or "
            "exceeded; "
            + "; ".join(failures)
        )

    decisions = {
        scope: _calibrated_decision(
            analytic,
            scale=chosen_scales[scope],
            iterations=brackets[scope].iterations,
            pre_error=pre_errors[scope],
            post_error=post_errors[scope],
            acceptance_tolerance=(
                storage_rounding_tolerance
                if scope in fallback_scopes
                else matching_tolerance
            ),
            storage_rounding_fallback=scope in fallback_scopes,
            strict_calibration_exhausted=scope in fallback_scopes,
            calibration_outcome=(
                "unmatchable"
                if analytic.unmatchable
                else (
                    "fp32_storage_lattice_fallback"
                    if scope in fallback_scopes
                    else "strict"
                )
            ),
        )
        for scope, analytic in analytic_decisions.items()
    }
    return _CalibrationResult(
        decisions=decisions,
        budgets=dict(realized_budgets),
        payload=payload,
    )


@torch.no_grad()
def apply_matched_deltas(
    before: Mapping[str, torch.Tensor],
    raw_deltas: Mapping[str, torch.Tensor],
    scope_map: ScopeMap,
    decisions: Mapping[str, MatchDecision],
    *,
    level: str,
) -> TensorMap:
    """Restore pre-step Adam parameters and apply the matched raw deltas.

    Optimizer state is deliberately untouched.  All scales are validated before
    any parameter is mutated, so a non-finite scale cannot partially apply.
    The one scope scalar is evaluated in fp64 and the resulting parameter is
    cast once to its native storage dtype.  This avoids an additional fp32
    multiply/add rounding lattice without changing the measured raw Adam
    direction or introducing coordinate-wise corrections.
    """

    named = dict(scope_map.named_parameters)
    _assert_same_names(named, before, "parameters", "before snapshot")
    _assert_same_names(named, raw_deltas, "parameters", "raw deltas")
    expected_scopes = set(scope_map.groups(level))
    if set(decisions) != expected_scopes:
        raise InstrumentationError(
            f"Decisions do not cover exactly the {level} scopes: "
            f"missing={sorted(expected_scopes - set(decisions))}, "
            f"extra={sorted(set(decisions) - expected_scopes)}"
        )
    for scope, decision in decisions.items():
        if not math.isfinite(decision.scale):
            raise NonFiniteMatchError(
                f"Refusing non-finite matching scale {decision.scale!r} "
                f"for scope {scope!r}."
            )

    for name, parameter in named.items():
        decision = decisions[scope_map.scope_for(name, level)]
        # Preserve the scientific operation ``before + scale * raw_delta``:
        # one scalar applies to the whole requested scope.  Computing that
        # expression in native fp32 rounds both the product and the sum before
        # storage, and on small block budgets can make Python-float scalar
        # refinements indistinguishable.  A double intermediate defers rounding
        # until this single copy back to the parameter's native dtype.
        candidate = raw_deltas[name].to(dtype=torch.float64)
        candidate.mul_(decision.scale)
        candidate.add_(before[name].to(dtype=torch.float64))
        parameter.copy_(candidate)
    return parameter_deltas(before, scope_map)


def _validate_optimizer_coverage(
    optimizer: Optimizer,
    named_parameters: Sequence[Tuple[str, nn.Parameter]],
) -> None:
    expected = {id(parameter) for _, parameter in named_parameters}
    actual_list = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    actual = {id(parameter) for parameter in actual_list}
    if len(actual) != len(actual_list):
        raise InstrumentationError("Optimizer contains a parameter more than once.")
    if expected != actual:
        raise InstrumentationError(
            "Instrumentation parameters do not exactly match optimizer parameters."
        )


@dataclass
class StepMeasurement:
    """Actual before/after tensors and delta from one native optimizer step."""

    before: TensorMap
    deltas: TensorMap
    after: TensorMap
    optimizer_return: object = None


def native_optimizer_step(
    optimizer: Optimizer,
    source: NamedParameterSource,
    *,
    closure=None,
) -> StepMeasurement:
    """Execute exactly one native optimizer step and record its actual delta."""

    named = _named_parameters(source, only_trainable=True)
    _validate_optimizer_coverage(optimizer, named)
    before = snapshot_parameters(named)
    optimizer_return = optimizer.step(closure)
    deltas = parameter_deltas(before, named)
    after = snapshot_parameters(named)
    return StepMeasurement(before, deltas, after, optimizer_return)


def _temporary_native_step(
    optimizer: Optimizer,
    source: NamedParameterSource,
) -> StepMeasurement:
    """Measure a native step, then restore parameters and optimizer state."""

    named = _named_parameters(source, only_trainable=True)
    _validate_optimizer_coverage(optimizer, named)
    before = snapshot_parameters(named)
    state = copy.deepcopy(optimizer.state_dict())
    try:
        optimizer.step()
        deltas = parameter_deltas(before, named)
        after = snapshot_parameters(named)
        result = StepMeasurement(before, deltas, after)
    finally:
        restore_parameters(named, before)
        optimizer.load_state_dict(state)
    return result


@dataclass
class M0StepResult:
    """Measurements and persistent LR produced by Adam-M0's first step."""

    raw_adam_step: StepMeasurement
    reference_sgd_step: StepMeasurement
    applied_adam_step: StepMeasurement
    raw_adam_budget: ScopeBudget
    reference_sgd_budget: ScopeBudget
    applied_adam_budget: ScopeBudget
    decision: MatchDecision
    nominal_lrs: Tuple[float, ...]
    effective_lrs: Tuple[float, ...]


@dataclass
class P5M0StepResult:
    """Exact P5 first-step match without a second persistent Adam advance."""

    raw_adam_step: StepMeasurement
    reference_sgd_step: StepMeasurement
    applied_adam_step: StepMeasurement
    raw_adam_budget: ScopeBudget
    reference_sgd_budget: ScopeBudget
    applied_adam_budget: ScopeBudget
    analytic_gamma_0: float
    gamma_0: float
    native_calibration_evaluations: int
    matched_learning_rate: float
    relative_matching_error: float


def _require_native_adam(optimizer: Optimizer) -> None:
    if type(optimizer) is not Adam:
        raise TypeError(
            "P1 Adam matching requires exactly torch.optim.Adam; "
            f"got {type(optimizer)!r}."
        )


def _optimizer_has_advanced(optimizer: Optimizer) -> bool:
    for state in optimizer.state.values():
        step = state.get("step", 0)
        if isinstance(step, torch.Tensor):
            if float(step.item()) != 0.0:
                return True
        elif step != 0:
            return True
    return False


def adam_m0_first_step(
    adam_optimizer: Adam,
    source: NamedParameterSource,
    *,
    reference_sgd_lr: float = 1e-4,
    epsilon_budget: float = 1e-12,
    matching_tolerance: float = 1e-5,
    storage_rounding_tolerance: float = 1e-4,
    max_calibration_iterations: int = 64,
) -> M0StepResult:
    """Perform Adam-M0's exact one-time first optimizer step.

    A temporary native Adam step and a temporary native zero-momentum SGD step
    measure the two proposals from the same parameters and gradients.  Native
    temporary steps then calibrate the scalar LR against the *realized* fp32
    parameter change.  Every temporary mutation and optimizer-state advance is
    restored.  Adam's group LRs are permanently multiplied by the calibrated
    scale and exactly one real native Adam step is executed.  Consequently Adam
    moment state advances once, not twice.
    """

    _require_native_adam(adam_optimizer)
    if getattr(adam_optimizer, "_p1_m0_complete", False):
        raise InstrumentationError("Adam-M0 first-step matching was already performed.")
    if _optimizer_has_advanced(adam_optimizer):
        raise InstrumentationError("Adam-M0 must be called before Adam has advanced.")
    if reference_sgd_lr < 0.0 or not math.isfinite(reference_sgd_lr):
        raise ValueError("reference_sgd_lr must be finite and non-negative.")

    named = _named_parameters(source, only_trainable=True)
    _validate_optimizer_coverage(adam_optimizer, named)
    nominal_lrs = tuple(float(group["lr"]) for group in adam_optimizer.param_groups)
    initial_parameters = snapshot_parameters(named)
    initial_optimizer_state = copy.deepcopy(adam_optimizer.state_dict())

    raw_adam_step = _temporary_native_step(adam_optimizer, named)
    reference_optimizer = SGD(
        [parameter for _, parameter in named],
        lr=reference_sgd_lr,
        momentum=0.0,
        weight_decay=0.0,
    )
    reference_sgd_step = _temporary_native_step(reference_optimizer, named)
    all_names = tuple(name for name, _ in named)
    raw_budget = _scope_budget(
        raw_adam_step.before, raw_adam_step.deltas, all_names
    )
    reference_budget = _scope_budget(
        reference_sgd_step.before, reference_sgd_step.deltas, all_names
    )
    analytic_decision = matching_scale(
        reference_budget.relative_budget,
        raw_budget.relative_budget,
        epsilon_budget=epsilon_budget,
        scope=GLOBAL_SCOPE,
    )
    try:
        def evaluate_scale(
            scales: Mapping[str, float],
        ) -> Tuple[Mapping[str, ScopeBudget], StepMeasurement]:
            scale = scales[GLOBAL_SCOPE]
            for group, nominal_lr in zip(
                adam_optimizer.param_groups, nominal_lrs
            ):
                group["lr"] = nominal_lr * scale
            candidate_step = _temporary_native_step(
                adam_optimizer, named
            )
            candidate_budget = _scope_budget(
                candidate_step.before,
                candidate_step.deltas,
                all_names,
            )
            return {GLOBAL_SCOPE: candidate_budget}, candidate_step

        calibration = _storage_aware_calibration(
            {GLOBAL_SCOPE: analytic_decision},
            {GLOBAL_SCOPE: reference_budget},
            evaluate_scale,
            epsilon_budget=epsilon_budget,
            matching_tolerance=matching_tolerance,
            storage_rounding_tolerance=storage_rounding_tolerance,
            max_calibration_iterations=max_calibration_iterations,
        )
        decision = calibration.decisions[GLOBAL_SCOPE]
        for group, nominal_lr in zip(
            adam_optimizer.param_groups, nominal_lrs
        ):
            group["lr"] = nominal_lr * decision.scale
        applied_step = native_optimizer_step(adam_optimizer, named)
        applied_budget = _scope_budget(
            applied_step.before, applied_step.deltas, all_names
        )
        final_error = _realized_matching_error(
            applied_budget.relative_budget,
            reference_budget.relative_budget,
            decision,
            epsilon_budget=epsilon_budget,
        )
        if (
            not decision.both_zero
            and not decision.unmatchable
            and (
                not math.isfinite(final_error)
                or final_error > decision.matching_acceptance_tolerance
            )
        ):
            raise PrecisionMatchError(
                "The final native Adam-M0 step did not reproduce its calibrated "
                f"budget: error={final_error!r}, "
                "tolerance="
                f"{decision.matching_acceptance_tolerance!r}."
            )
        decision = replace(
            decision,
            postcalibration_relative_error=final_error,
        )
    except Exception:
        restore_parameters(named, initial_parameters)
        adam_optimizer.load_state_dict(initial_optimizer_state)
        raise
    setattr(adam_optimizer, "_p1_m0_complete", True)
    effective_lrs = tuple(float(group["lr"]) for group in adam_optimizer.param_groups)
    return M0StepResult(
        raw_adam_step=raw_adam_step,
        reference_sgd_step=reference_sgd_step,
        applied_adam_step=applied_step,
        raw_adam_budget=raw_budget,
        reference_sgd_budget=reference_budget,
        applied_adam_budget=applied_budget,
        decision=decision,
        nominal_lrs=nominal_lrs,
        effective_lrs=effective_lrs,
    )


def p5_adam_m0_first_step(
    adam_optimizer: Adam,
    scope_map: ScopeMap,
    *,
    reference_sgd_lr: float = 1e-4,
    epsilon_budget: float = 1e-12,
    matching_tolerance: float = 1e-5,
) -> P5M0StepResult:
    """Apply P5 Adam-M0 using exact temporary native PyTorch proposals.

    The temporary raw Adam step is restored to the empty pre-step state. The
    final matched call is therefore the only persistent moment and parameter
    update, and every Adam ``step`` tensor must equal one on return.
    """

    _require_native_adam(adam_optimizer)
    if getattr(adam_optimizer, "_p5_m0_complete", False):
        raise InstrumentationError("P5 Adam-M0 first-step matching already ran")
    if _optimizer_has_advanced(adam_optimizer) or adam_optimizer.state:
        raise InstrumentationError("P5 Adam-M0 requires pristine Adam state")
    named = scope_map.named_parameters
    initial_parameters = snapshot_parameters(scope_map)
    initial_state = copy.deepcopy(adam_optimizer.state_dict())
    nominal_lrs = tuple(float(group["lr"]) for group in adam_optimizer.param_groups)
    if len(set(nominal_lrs)) != 1:
        raise InstrumentationError("P5 Adam-M0 expects one nominal learning rate")
    try:
        raw_step = _temporary_native_step(adam_optimizer, scope_map)
        reference_optimizer = SGD(
            [parameter for _, parameter in named],
            lr=reference_sgd_lr,
            momentum=0.0,
            weight_decay=0.0,
            dampening=0.0,
            nesterov=False,
        )
        reference_step = _temporary_native_step(reference_optimizer, scope_map)
        raw_budget = compute_scope_budgets(
            raw_step.before,
            raw_step.deltas,
            scope_map,
            levels=("global",),
        )[GLOBAL_SCOPE]
        reference_budget = compute_scope_budgets(
            reference_step.before,
            reference_step.deltas,
            scope_map,
            levels=("global",),
        )[GLOBAL_SCOPE]
        analytic_gamma_0 = reference_budget.relative_budget / (
            raw_budget.relative_budget + epsilon_budget
        )
        if not math.isfinite(analytic_gamma_0) or analytic_gamma_0 < 0:
            raise NonFiniteMatchError(
                f"Invalid P5 analytic gamma_0: {analytic_gamma_0!r}"
            )
        # The analytic ratio is exact in real arithmetic, but adding an Adam
        # delta to 170M fp32 parameters introduces a measurable storage
        # quantization error. Calibrate the one permanent LR against exact
        # temporary native Adam steps. Each proposal restores the pristine
        # optimizer state, so only the final step advances moments.
        calibration_evaluations = 0

        def realized_budget(candidate_gamma: float) -> ScopeBudget:
            nonlocal calibration_evaluations
            for group, nominal_lr in zip(adam_optimizer.param_groups, nominal_lrs):
                group["lr"] = nominal_lr * candidate_gamma
            candidate_step = _temporary_native_step(adam_optimizer, scope_map)
            candidate_budget = compute_scope_budgets(
                candidate_step.before,
                candidate_step.deltas,
                scope_map,
                levels=("global",),
            )[GLOBAL_SCOPE]
            del candidate_step
            calibration_evaluations += 1
            if not math.isfinite(candidate_budget.relative_budget):
                raise NonFiniteMatchError(
                    "Non-finite realized Adam-M0 candidate budget"
                )
            return candidate_budget

        target_budget = reference_budget.relative_budget
        if not math.isfinite(target_budget) or target_budget <= 0:
            raise NonFiniteMatchError(
                f"Invalid SGD reference budget: {target_budget!r}"
            )
        low_gamma = 0.0
        high_gamma = max(2.0 * analytic_gamma_0, 1e-16)
        high_budget = realized_budget(high_gamma).relative_budget
        for _ in range(8):
            if high_budget >= target_budget:
                break
            low_gamma = high_gamma
            high_gamma *= 2.0
            high_budget = realized_budget(high_gamma).relative_budget
        else:
            raise PrecisionMatchError(
                "P5 Adam-M0 could not bracket the SGD reference budget"
            )
        best_gamma = high_gamma
        best_error = abs(high_budget - target_budget) / (
            target_budget + epsilon_budget
        )
        for _ in range(32):
            candidate_gamma = (low_gamma + high_gamma) / 2.0
            candidate_budget = realized_budget(
                candidate_gamma
            ).relative_budget
            candidate_error = abs(candidate_budget - target_budget) / (
                target_budget + epsilon_budget
            )
            if candidate_error < best_error:
                best_gamma = candidate_gamma
                best_error = candidate_error
            if candidate_error <= matching_tolerance:
                break
            if candidate_budget < target_budget:
                low_gamma = candidate_gamma
            else:
                high_gamma = candidate_gamma
        if best_error > matching_tolerance:
            raise PrecisionMatchError(
                "P5 Adam-M0 bracketed native LR calibration did not meet "
                f"{matching_tolerance:.8e}; best error={best_error:.8e}"
            )
        gamma_0 = best_gamma
        for group, nominal_lr in zip(adam_optimizer.param_groups, nominal_lrs):
            group["lr"] = nominal_lr * gamma_0
        applied_step = native_optimizer_step(adam_optimizer, scope_map)
        applied_budget = compute_scope_budgets(
            applied_step.before,
            applied_step.deltas,
            scope_map,
            levels=("global",),
        )[GLOBAL_SCOPE]
        relative_error = abs(
            applied_budget.relative_budget - reference_budget.relative_budget
        ) / (reference_budget.relative_budget + epsilon_budget)
        if not math.isfinite(relative_error) or relative_error > matching_tolerance:
            raise PrecisionMatchError(
                "P5 Adam-M0 first-step matching error "
                f"{relative_error:.8e} exceeds {matching_tolerance:.8e}"
            )
        for state in adam_optimizer.state.values():
            step = state.get("step")
            value = float(step.item()) if torch.is_tensor(step) else float(step)
            if value != 1.0:
                raise InstrumentationError(
                    f"Adam state advanced {value} times instead of exactly once"
                )
    except Exception:
        restore_parameters(scope_map, initial_parameters)
        adam_optimizer.load_state_dict(initial_state)
        raise
    setattr(adam_optimizer, "_p5_m0_complete", True)
    return P5M0StepResult(
        raw_adam_step=raw_step,
        reference_sgd_step=reference_step,
        applied_adam_step=applied_step,
        raw_adam_budget=raw_budget,
        reference_sgd_budget=reference_budget,
        applied_adam_budget=applied_budget,
        analytic_gamma_0=analytic_gamma_0,
        gamma_0=gamma_0,
        native_calibration_evaluations=calibration_evaluations,
        matched_learning_rate=float(adam_optimizer.param_groups[0]["lr"]),
        relative_matching_error=relative_error,
    )


@contextlib.contextmanager
def preserve_rng_and_mode(model: nn.Module):
    """Restore Python, NumPy, CPU/CUDA Torch RNG and train/eval mode."""

    import random

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    training = bool(model.training)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        model.train(training)


def assert_training_batch_source(source: str) -> None:
    """Prevent full-reference diagnostics from entering an optimizer update."""

    if source != "support":
        raise InstrumentationError(
            f"Only support batches may reach optimizer.step(); got source={source!r}"
        )


def compute_gradient_moments(
    adapter: nn.Module,
    batches: Sequence[Sequence[Tuple[str, int]]],
    transform,
    text_inputs: Mapping[str, torch.Tensor],
    scope_map: ScopeMap,
    device: str,
) -> Tuple[TensorMap, TensorMap]:
    """Compute equal-weight ``m`` and ``v`` from deterministic batch gradients."""

    if not batches:
        raise ValueError("Diagnostic batches cannot be empty")
    accum_m = {
        name: torch.zeros_like(parameter, dtype=torch.float32, device=parameter.device)
        for name, parameter in scope_map.named_parameters
    }
    accum_v = {
        name: torch.zeros_like(parameter, dtype=torch.float32, device=parameter.device)
        for name, parameter in scope_map.named_parameters
    }
    weight = 1.0 / len(batches)
    with preserve_rng_and_mode(adapter):
        adapter.eval()
        for batch in batches:
            images = []
            labels = []
            for path, label in batch:
                with Image.open(path) as handle:
                    images.append(transform(handle.convert("RGB")))
                labels.append(label)
            pixels = torch.stack(images).to(device)
            targets = torch.tensor(labels, dtype=torch.long, device=device)
            adapter.zero_grad(set_to_none=True)
            image_embeddings = adapter.encode_image(pixels)
            text_embeddings = adapter.encode_text(text_inputs)
            logits = adapter.classification_logits(image_embeddings, text_embeddings)
            loss = F.cross_entropy(logits.float(), targets, reduction="mean")
            loss.backward()
            for name, parameter in scope_map.named_parameters:
                if parameter.grad is None:
                    raise InstrumentationError(
                        f"Diagnostic produced no gradient for {name!r}"
                    )
                gradient = parameter.grad.detach().float()
                accum_m[name].add_(gradient, alpha=weight)
                accum_v[name].addcmul_(gradient, gradient, value=weight)
        adapter.zero_grad(set_to_none=True)
    return (
        {name: value.cpu() for name, value in accum_m.items()},
        {name: value.cpu() for name, value in accum_v.items()},
    )


_MT_NATIVE_SHADOW_COUNT = 2
_MT_NATIVE_WORKSPACE_ATTRIBUTE = "_p1_mt_native_adam_lr_workspace"


@dataclass
class _NativeAdamLRWorkspace:
    """Bounded native-Adam search workspace synchronized with the main Adam.

    ``anchor`` retains the exact Adam state at time ``t`` while ``searcher`` is
    restored from it in place before each native effective-LR evaluation.
    After a scalar has been selected, ``anchor`` advances once at nominal LR.
    Adam's moment update is independent of LR, so main, searcher, and anchor
    then hold bitwise-identical time-``t+1`` states.  The two shadows persist
    across steps; scalar-search depth therefore changes runtime, not memory.
    """

    level: str
    parameter_ids: Tuple[int, ...]
    nominal_lr: float
    scopes: Tuple[str, ...]
    scope_names: Dict[str, Tuple[str, ...]]
    searcher: Adam
    anchor: Adam


def _clone_adam_state_value(value):
    """Clone one Adam state value for one-time workspace initialization."""

    if isinstance(value, torch.Tensor):
        return value.detach().clone(memory_format=torch.preserve_format)
    return copy.deepcopy(value)


def _initialize_shadow_state_once(
    source: Adam,
    shadow: Adam,
    parameters: Sequence[nn.Parameter],
) -> None:
    """Initialize one persistent shadow without using ``state_dict`` copies."""

    shadow.state.clear()
    for parameter in parameters:
        source_state = source.state.get(parameter)
        if not source_state:
            continue
        shadow.state[parameter] = {
            key: _clone_adam_state_value(value)
            for key, value in source_state.items()
        }


def _restore_searcher_state_from_anchor(
    searcher: Adam,
    anchor: Adam,
    named_parameters: Sequence[Tuple[str, nn.Parameter]],
) -> None:
    """Restore the searcher to anchor time ``t`` without allocating a copy.

    Adam initializes parameter state lazily. On a parameter's first update,
    the untouched anchor has no state while a previously evaluated searcher
    owns the usual zero-at-time-zero buffers. Zeroing those retained buffers
    is exactly equivalent to Adam's lazy initialization and avoids allocator
    churn during a first-step scalar search.
    """

    for name, parameter in named_parameters:
        anchor_state = anchor.state.get(parameter, {})
        searcher_state = searcher.state.get(parameter, {})
        if anchor_state:
            if set(anchor_state) != set(searcher_state):
                raise InstrumentationError(
                    "Native Adam searcher/anchor state keys diverged for "
                    f"{name!r}: anchor={sorted(anchor_state)}, "
                    f"searcher={sorted(searcher_state)}"
                )
            for key, anchor_value in anchor_state.items():
                searcher_value = searcher_state[key]
                if isinstance(anchor_value, torch.Tensor):
                    if not isinstance(searcher_value, torch.Tensor):
                        raise InstrumentationError(
                            "Native Adam searcher/anchor state types diverged "
                            f"for {name!r} state {key!r}."
                        )
                    searcher_value.copy_(anchor_value)
                else:
                    searcher_state[key] = copy.deepcopy(anchor_value)
            continue

        for key, value in searcher_state.items():
            if key not in {
                "step",
                "exp_avg",
                "exp_avg_sq",
                "max_exp_avg_sq",
            }:
                raise InstrumentationError(
                    "Cannot restore unknown lazy Adam state "
                    f"{key!r} for {name!r}."
                )
            if isinstance(value, torch.Tensor):
                value.zero_()
            elif key == "step" and isinstance(value, (int, float)):
                searcher_state[key] = type(value)(0)
            else:
                raise InstrumentationError(
                    "Cannot zero lazy Adam state "
                    f"{key!r} for {name!r}: {type(value)!r}."
                )


def _state_values_are_identical(left, right) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.device == right.device
            and torch.equal(left, right)
        )
    return left == right


def _assert_adam_states_synchronized(
    reference: Adam,
    candidate: Adam,
    named_parameters: Sequence[Tuple[str, nn.Parameter]],
) -> None:
    """Fail if a persistent candidate no longer mirrors native Adam moments."""

    for name, parameter in named_parameters:
        reference_state = reference.state.get(parameter, {})
        candidate_state = candidate.state.get(parameter, {})
        if set(reference_state) != set(candidate_state):
            raise InstrumentationError(
                "Native Adam LR candidate state keys diverged for "
                f"{name!r}: reference={sorted(reference_state)}, "
                f"candidate={sorted(candidate_state)}"
            )
        for key in reference_state:
            if not _state_values_are_identical(
                reference_state[key], candidate_state[key]
            ):
                raise InstrumentationError(
                    "Native Adam LR candidate moments diverged for "
                    f"{name!r} state {key!r}."
                )


def _optimizer_step_value(
    optimizer: Adam,
    parameter: nn.Parameter,
) -> Optional[float]:
    state = optimizer.state.get(parameter, {})
    if "step" not in state:
        return None
    value = state["step"]
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


def _assert_main_adam_advanced_once(
    optimizer: Adam,
    named_parameters: Sequence[Tuple[str, nn.Parameter]],
    before_steps: Mapping[int, Optional[float]],
) -> None:
    for name, parameter in named_parameters:
        before = before_steps[id(parameter)]
        after = _optimizer_step_value(optimizer, parameter)
        if parameter.grad is None:
            expected = before
        else:
            expected = 1.0 if before is None else before + 1.0
        if after != expected:
            raise InstrumentationError(
                "Main Adam state did not advance exactly once for "
                f"{name!r}: before={before!r}, after={after!r}, "
                f"expected={expected!r}."
            )


def _build_native_adam_lr_workspace(
    adam_optimizer: Adam,
    scope_map: ScopeMap,
    *,
    level: str,
) -> _NativeAdamLRWorkspace:
    """Create the persistent searcher and untouched time-``t`` anchor."""

    if len(adam_optimizer.param_groups) != 1:
        raise InstrumentationError(
            "Native Adam LR realization requires Plain's single Adam "
            "parameter group."
        )
    main_group = adam_optimizer.param_groups[0]
    nominal_lr = float(main_group["lr"])
    if not math.isfinite(nominal_lr) or nominal_lr < 0.0:
        raise NonFiniteMatchError(
            f"Invalid nominal Adam LR for Mt realization: {nominal_lr!r}."
        )
    named = scope_map.named_parameters
    parameters = dict(named)
    scope_names = scope_map.groups(level)
    group_options = {
        key: value
        for key, value in main_group.items()
        if key not in {"params", "lr"}
    }
    shadows = []
    for _ in range(_MT_NATIVE_SHADOW_COUNT):
        scoped_groups = []
        for scope, names in scope_names.items():
            group = dict(group_options)
            group["params"] = [parameters[name] for name in names]
            group["lr"] = nominal_lr
            group["_p1_matching_scope"] = scope
            scoped_groups.append(group)
        shadow = Adam(scoped_groups, **dict(adam_optimizer.defaults))
        _validate_optimizer_coverage(shadow, named)
        _initialize_shadow_state_once(
            adam_optimizer, shadow, scope_map.parameters
        )
        _assert_adam_states_synchronized(
            adam_optimizer, shadow, named
        )
        shadows.append(shadow)
    return _NativeAdamLRWorkspace(
        level=level,
        parameter_ids=tuple(id(parameter) for parameter in scope_map.parameters),
        nominal_lr=nominal_lr,
        scopes=tuple(scope_names),
        scope_names=dict(scope_names),
        searcher=shadows[0],
        anchor=shadows[1],
    )


def _native_adam_lr_workspace(
    adam_optimizer: Adam,
    scope_map: ScopeMap,
    *,
    level: str,
) -> _NativeAdamLRWorkspace:
    cached = getattr(adam_optimizer, _MT_NATIVE_WORKSPACE_ATTRIBUTE, None)
    if cached is None:
        cached = _build_native_adam_lr_workspace(
            adam_optimizer,
            scope_map,
            level=level,
        )
        setattr(adam_optimizer, _MT_NATIVE_WORKSPACE_ATTRIBUTE, cached)
    expected_ids = tuple(id(parameter) for parameter in scope_map.parameters)
    if cached.level != level or cached.parameter_ids != expected_ids:
        raise InstrumentationError(
            "A native Adam LR workspace cannot be reused with a different "
            "matching level or parameter mapping."
        )
    if len(adam_optimizer.param_groups) != 1:
        raise InstrumentationError(
            "The main Adam optimizer no longer has Plain's single group."
        )
    current_lr = float(adam_optimizer.param_groups[0]["lr"])
    if current_lr != cached.nominal_lr:
        raise InstrumentationError(
            "The main Adam LR changed between Mt steps: "
            f"nominal={cached.nominal_lr!r}, current={current_lr!r}."
        )
    for candidate in (cached.searcher, cached.anchor):
        _assert_adam_states_synchronized(
            adam_optimizer, candidate, scope_map.named_parameters
        )
    return cached


def _set_native_candidate_lrs(
    candidate: Adam,
    workspace: _NativeAdamLRWorkspace,
    scales: Mapping[str, float],
) -> None:
    if set(scales) != set(workspace.scopes):
        raise InstrumentationError(
            "Native Adam LR candidate scales do not cover the matching scopes."
        )
    if len(candidate.param_groups) != len(workspace.scopes):
        raise InstrumentationError(
            "Native Adam LR candidate group coverage changed."
        )
    for group, scope in zip(candidate.param_groups, workspace.scopes):
        scale = float(scales[scope])
        effective_lr = workspace.nominal_lr * scale
        if (
            not math.isfinite(scale)
            or scale < 0.0
            or not math.isfinite(effective_lr)
            or effective_lr < 0.0
        ):
            raise NonFiniteMatchError(
                f"Invalid native Adam LR scale {scale!r} for scope {scope!r}."
            )
        group["lr"] = effective_lr


def _native_adam_lr_realization(
    adam_optimizer: Adam,
    adam_scope_map: ScopeMap,
    raw_step: StepMeasurement,
    raw_budgets: Mapping[str, ScopeBudget],
    reference_budgets: Mapping[str, ScopeBudget],
    analytic_decisions: Mapping[str, MatchDecision],
    workspace: _NativeAdamLRWorkspace,
    *,
    epsilon_budget: float,
    matching_tolerance: float,
    storage_rounding_tolerance: float,
    max_calibration_iterations: int,
) -> _CalibrationResult[TensorMap]:
    """Search native effective LRs with two persistent shadow optimizers.

    The searcher is restored in place from the untouched time-``t`` anchor for
    every evaluator call. The safeguarded monotone scalar calibrator supplies
    ratio/bracket candidates and alone decides whether adjacent binary64
    endpoints prove the storage lattice exhausted. An iteration cap can
    therefore never authorize relaxed acceptance.
    """

    _assert_same_names(
        analytic_decisions,
        reference_budgets,
        "matching decisions",
        "reference budgets",
    )
    _assert_same_names(
        analytic_decisions,
        raw_budgets,
        "matching decisions",
        "raw budgets",
    )

    searcher = workspace.searcher
    anchor = workspace.anchor
    unit_scales = {scope: 1.0 for scope in workspace.scopes}

    def evaluate_scales(
        scales: Mapping[str, float],
    ) -> Tuple[Mapping[str, ScopeBudget], None]:
        restore_parameters(adam_scope_map, raw_step.before)
        _restore_searcher_state_from_anchor(
            searcher,
            anchor,
            adam_scope_map.named_parameters,
        )
        _set_native_candidate_lrs(searcher, workspace, scales)
        searcher.step()
        candidate_deltas = parameter_deltas(
            raw_step.before, adam_scope_map
        )
        candidate_budgets = compute_scope_budgets(
            raw_step.before,
            candidate_deltas,
            adam_scope_map,
            levels=(workspace.level,),
        )
        return candidate_budgets, None

    try:
        calibration = _storage_aware_calibration(
            analytic_decisions,
            reference_budgets,
            evaluate_scales,
            epsilon_budget=epsilon_budget,
            matching_tolerance=matching_tolerance,
            storage_rounding_tolerance=storage_rounding_tolerance,
            max_calibration_iterations=max_calibration_iterations,
        )
    except Exception:
        # A failed run still retains exactly one main Adam state advance.
        # Re-synchronize the bounded workspace so failure inspection cannot
        # observe a search-depth-dependent moment state.
        restore_parameters(adam_scope_map, raw_step.before)
        _restore_searcher_state_from_anchor(
            searcher,
            anchor,
            adam_scope_map.named_parameters,
        )
        _set_native_candidate_lrs(searcher, workspace, unit_scales)
        searcher.step()
        restore_parameters(adam_scope_map, raw_step.before)
        _set_native_candidate_lrs(anchor, workspace, unit_scales)
        anchor.step()
        for shadow in (searcher, anchor):
            _assert_adam_states_synchronized(
                adam_optimizer, shadow, adam_scope_map.named_parameters
            )
        restore_parameters(adam_scope_map, raw_step.before)
        raise

    # The safeguarded calibrator always leaves the exact final evaluator
    # tensor live: either the first strict candidate, the last candidate that
    # completed all scopes, or the explicit joint best-scalar re-evaluation.
    # Clone it once here instead of retaining one full-model payload per call.
    selected_after = snapshot_parameters(adam_scope_map)
    restore_parameters(adam_scope_map, raw_step.before)
    _set_native_candidate_lrs(anchor, workspace, unit_scales)
    anchor.step()
    for shadow in (searcher, anchor):
        _assert_adam_states_synchronized(
            adam_optimizer, shadow, adam_scope_map.named_parameters
        )

    restore_parameters(adam_scope_map, selected_after)
    applied_deltas = parameter_deltas(raw_step.before, adam_scope_map)
    applied_budgets = compute_scope_budgets(
        raw_step.before,
        applied_deltas,
        adam_scope_map,
        levels=(workspace.level,),
    )

    decisions = {}
    for scope, generic_decision in calibration.decisions.items():
        analytic = analytic_decisions[scope]
        error = _realized_matching_error(
            applied_budgets[scope].relative_budget,
            reference_budgets[scope].relative_budget,
            analytic,
            epsilon_budget=epsilon_budget,
        )
        if (
            not analytic.both_zero
            and not analytic.unmatchable
            and (
                not math.isfinite(error)
                or error > generic_decision.matching_acceptance_tolerance
            )
        ):
            raise PrecisionMatchError(
                "The selected native Adam effective-LR tensor did not retain "
                f"its calibrated budget for {scope!r}: error={error!r}, "
                "tolerance="
                f"{generic_decision.matching_acceptance_tolerance!r}."
            )
        expected_error = generic_decision.postcalibration_relative_error
        errors_agree = (
            math.isnan(error)
            and math.isnan(expected_error)
        ) or math.isclose(
            error,
            expected_error,
            rel_tol=1.0e-12,
            abs_tol=1.0e-18,
        )
        if not errors_agree:
            raise InstrumentationError(
                "The selected native Adam tensor disagrees with its "
                f"calibration payload for {scope!r}."
            )
        decisions[scope] = replace(
            generic_decision,
            postcalibration_relative_error=error,
            calibration_outcome=(
                "unmatchable"
                if analytic.unmatchable
                else (
                    "native_adam_lr_storage_fallback"
                    if generic_decision.storage_rounding_fallback
                    else "native_adam_lr_strict"
                )
            ),
        )
    return _CalibrationResult(
        decisions=decisions,
        budgets=applied_budgets,
        payload=applied_deltas,
    )


@dataclass
class MtStepResult:
    """Native raw and matched applied Adam measurements for one Mt step."""

    raw_adam_step: StepMeasurement
    applied_deltas: TensorMap
    raw_adam_budgets: Dict[str, ScopeBudget]
    reference_sgd_budgets: Dict[str, ScopeBudget]
    applied_adam_budgets: Dict[str, ScopeBudget]
    decisions: Dict[str, MatchDecision]
    level: str


def adam_mt_step(
    adam_optimizer: Adam,
    adam_scope_map: ScopeMap,
    reference_sgd_step: StepMeasurement,
    reference_scope_map: ScopeMap,
    *,
    match_level: str = "global",
    epsilon_budget: float = 1e-12,
    matching_tolerance: float = 1e-5,
    storage_rounding_tolerance: float = 1e-4,
    max_calibration_iterations: int = 64,
) -> MtStepResult:
    """Perform one native Adam step and realize matching via effective LRs.

    The caller first advances the paired native SGD trajectory and passes its
    :class:`StepMeasurement`.  The main Adam advances once at its nominal LR to
    produce the raw proposal. A persistent searcher is restored in place from
    an untouched time-``t`` anchor before every native effective-LR candidate
    step. The exact selected stored tensor is retained, then the anchor
    advances once at nominal LR. Main, searcher, and anchor moment tensors must
    be bitwise equal at time ``t+1``.

    The safeguarded search uses at most ``max_calibration_iterations``
    refinements. Reaching that cap is not evidence of storage-lattice
    exhaustion; relaxed acceptance is available only after adjacent binary64
    bracket endpoints prove that no scalar remains between them.
    """

    _require_native_adam(adam_optimizer)
    if max_calibration_iterations <= 0:
        raise ValueError("max_calibration_iterations must be positive.")
    if (
        not math.isfinite(matching_tolerance)
        or matching_tolerance < 0.0
    ):
        raise ValueError("matching_tolerance must be finite and non-negative.")
    if (
        not math.isfinite(storage_rounding_tolerance)
        or storage_rounding_tolerance < matching_tolerance
    ):
        raise ValueError(
            "storage_rounding_tolerance must be finite and no smaller than "
            "matching_tolerance."
        )
    if match_level not in {"global", "encoder", "block"}:
        raise ValueError(
            f"Unknown match_level {match_level!r}; expected global, encoder, or block."
        )
    if adam_scope_map.groups(match_level).keys() != reference_scope_map.groups(
        match_level
    ).keys():
        raise ScopeMappingError(
            "Adam and reference SGD maps do not expose identical matching scopes."
        )

    workspace = _native_adam_lr_workspace(
        adam_optimizer,
        adam_scope_map,
        level=match_level,
    )
    before_steps = {
        id(parameter): _optimizer_step_value(adam_optimizer, parameter)
        for parameter in adam_scope_map.parameters
    }
    raw_step = native_optimizer_step(adam_optimizer, adam_scope_map)
    _assert_main_adam_advanced_once(
        adam_optimizer,
        adam_scope_map.named_parameters,
        before_steps,
    )
    raw_budgets = compute_scope_budgets(
        raw_step.before,
        raw_step.deltas,
        adam_scope_map,
        levels=(match_level,),
    )
    reference_budgets = compute_scope_budgets(
        reference_sgd_step.before,
        reference_sgd_step.deltas,
        reference_scope_map,
        levels=(match_level,),
    )
    try:
        analytic_decisions = compute_matching_scales(
            reference_budgets,
            raw_budgets,
            epsilon_budget=epsilon_budget,
        )
        calibration = _native_adam_lr_realization(
            adam_optimizer,
            adam_scope_map,
            raw_step,
            raw_budgets,
            reference_budgets,
            analytic_decisions,
            workspace,
            epsilon_budget=epsilon_budget,
            matching_tolerance=matching_tolerance,
            storage_rounding_tolerance=storage_rounding_tolerance,
            max_calibration_iterations=max_calibration_iterations,
        )
        decisions = calibration.decisions
        applied_deltas = calibration.payload
        applied_budgets = calibration.budgets
    except Exception:
        # Leave no partially/raw-updated parameters in a failed run.  The native
        # Adam state advance is intentionally retained, matching Mt's invariant.
        restore_parameters(adam_scope_map, raw_step.before)
        raise

    return MtStepResult(
        raw_adam_step=raw_step,
        applied_deltas=applied_deltas,
        raw_adam_budgets=raw_budgets,
        reference_sgd_budgets=reference_budgets,
        applied_adam_budgets=applied_budgets,
        decisions=decisions,
        level=match_level,
    )


@dataclass(frozen=True)
class WarmupSchedule:
    """Exact zero-indexed P1 linear-warmup schedule."""

    base_lr: float
    total_steps: int
    warmup_fraction: float
    decay: str = "constant"
    final_lr_ratio: float = 0.01

    def __post_init__(self) -> None:
        if self.base_lr < 0.0 or not math.isfinite(self.base_lr):
            raise ValueError("base_lr must be finite and non-negative.")
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive.")
        if not 0.0 <= self.warmup_fraction <= 1.0:
            raise ValueError("warmup_fraction must lie in [0, 1].")
        if self.decay not in {"constant", "cosine"}:
            raise ValueError("decay must be 'constant' or 'cosine'.")
        if not 0.0 <= self.final_lr_ratio <= 1.0:
            raise ValueError("final_lr_ratio must lie in [0, 1].")

    @property
    def warmup_steps(self) -> int:
        return max(1, math.ceil(self.warmup_fraction * self.total_steps))

    def lr_at(self, step: int) -> float:
        """Return LR for zero-indexed optimizer step ``step``."""

        if not 0 <= step < self.total_steps:
            raise IndexError(
                f"step must be in [0, {self.total_steps}); got {step}."
            )
        warmup_steps = self.warmup_steps
        if step < warmup_steps:
            return self.base_lr * min((step + 1) / warmup_steps, 1.0)
        if self.decay == "constant":
            return self.base_lr

        post_warmup_steps = self.total_steps - warmup_steps
        if post_warmup_steps <= 0:
            return self.base_lr
        # Warmup's last step is the cosine's base-LR endpoint (progress 0);
        # the final optimizer step is progress 1 and therefore exactly 0.01x.
        progress = (step - warmup_steps + 1) / post_warmup_steps
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        multiplier = self.final_lr_ratio + (1.0 - self.final_lr_ratio) * cosine
        return self.base_lr * multiplier

    def sequence(self) -> Tuple[float, ...]:
        """Return all per-step LRs."""

        return tuple(self.lr_at(step) for step in range(self.total_steps))

    def apply(self, optimizer: Optimizer, step: int) -> float:
        """Set every optimizer parameter group to this step's LR."""

        learning_rate = self.lr_at(step)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        return learning_rate


def gradients_are_finite(source: NamedParameterSource) -> bool:
    """Return whether all present gradients are finite (missing grads are valid)."""

    for _, parameter in _named_parameters(source, only_trainable=True):
        gradient = parameter.grad
        if gradient is None:
            continue
        values = gradient.detach()
        if values.is_sparse:
            values = values.coalesce().values()
        if not bool(torch.isfinite(values).all().item()):
            return False
    return True


def amp_found_inf(scaler, optimizer: Optimizer) -> bool:
    """Read GradScaler's post-unscale found-inf state for one optimizer.

    PyTorch currently exposes this information only in GradScaler's optimizer
    bookkeeping.  The helper is isolated here so the training loop does not
    depend on that representation.  Gradient finiteness remains an independent
    public check.
    """

    states = getattr(scaler, "_per_optimizer_states", None)
    if states is None:
        return False
    state = states.get(id(optimizer))
    if not state:
        return False
    found = state.get("found_inf_per_device", {})
    return any(float(value.item()) != 0.0 for value in found.values())


@dataclass(frozen=True)
class SynchronizedStepDecision:
    """Whether paired trajectories may both advance."""

    should_step: bool
    step_skipped: bool
    nonfinite_trajectories: Tuple[str, ...]
    amp_skipped_trajectories: Tuple[str, ...]


def synchronized_step_decision(
    trajectories: Mapping[str, NamedParameterSource],
    *,
    amp_skip_flags: Optional[Mapping[str, bool]] = None,
) -> SynchronizedStepDecision:
    """Gate paired steps so one trajectory can never advance alone."""

    if not trajectories:
        raise ValueError("At least one trajectory is required.")
    nonfinite = tuple(
        name
        for name, parameters in trajectories.items()
        if not gradients_are_finite(parameters)
    )
    flags = amp_skip_flags or {}
    unknown = set(flags) - set(trajectories)
    if unknown:
        raise ValueError(f"AMP flags contain unknown trajectories: {sorted(unknown)}")
    amp_skipped = tuple(name for name in trajectories if bool(flags.get(name, False)))
    skipped = bool(nonfinite or amp_skipped)
    return SynchronizedStepDecision(
        should_step=not skipped,
        step_skipped=skipped,
        nonfinite_trajectories=nonfinite,
        amp_skipped_trajectories=amp_skipped,
    )


def synchronized_amp_decision(
    trajectories: Mapping[str, NamedParameterSource],
    scaler,
    optimizers: Mapping[str, Optimizer],
) -> SynchronizedStepDecision:
    """Combine explicit gradient and GradScaler skip checks after unscale."""

    if set(trajectories) != set(optimizers):
        raise ValueError("Trajectory and optimizer names must match.")
    flags = {
        name: amp_found_inf(scaler, optimizer)
        for name, optimizer in optimizers.items()
    }
    return synchronized_step_decision(trajectories, amp_skip_flags=flags)


def advance_scaler_after_synchronized_skip(scaler) -> None:
    """Safely update a scaler after both paired optimizer steps were withheld.

    Call this after both optimizers have been unscaled/checked.  It updates AMP's
    dynamic scale while deliberately advancing neither optimizer.
    """

    scaler.update()


@dataclass(frozen=True)
class ParameterDrift:
    """Parameter displacement relative to the initial parameter norm."""

    initial_l2: float
    delta_l2: float
    relative_drift: float


def compute_parameter_drift(
    initial: Mapping[str, torch.Tensor],
    scope_map: ScopeMap,
    *,
    levels: Sequence[str] = ("global", "encoder", "block"),
    include_empty: bool = False,
) -> Dict[str, ParameterDrift]:
    """Measure fp32 parameter drift globally, per encoder, and per block."""

    current = snapshot_parameters(scope_map)
    _assert_same_names(current, initial, "current parameters", "initial snapshot")
    # Form the displacement itself in fp32, not merely its reduction.  CLIP's
    # trainable weights are commonly fp16 on CUDA, and subtracting two saved
    # fp16 snapshots first would unnecessarily round the diagnostic.
    deltas = {
        name: current[name].detach().to(dtype=torch.float32)
        - initial[name].detach().to(dtype=torch.float32)
        for name in current
    }
    budgets = compute_scope_budgets(
        initial,
        deltas,
        scope_map,
        levels=levels,
        include_empty=include_empty,
    )
    return {
        scope: ParameterDrift(
            initial_l2=value.parameter_l2,
            delta_l2=value.update_l2,
            relative_drift=value.relative_budget,
        )
        for scope, value in budgets.items()
    }


__all__ = [
    "GLOBAL_SCOPE",
    "ENCODER_SCOPES",
    "InstrumentationError",
    "ScopeMappingError",
    "NonFiniteMatchError",
    "PrecisionMatchError",
    "ScopeEntry",
    "ScopeMap",
    "ScopeBudget",
    "MatchDecision",
    "StepMeasurement",
    "M0StepResult",
    "P5M0StepResult",
    "MtStepResult",
    "WarmupSchedule",
    "SynchronizedStepDecision",
    "ParameterDrift",
    "build_siglip_scope_map",
    "build_clip_scope_map",
    "snapshot_parameters",
    "restore_parameters",
    "parameter_deltas",
    "fp32_l2_norm",
    "compute_scope_budgets",
    "matching_scale",
    "compute_matching_scales",
    "apply_matched_deltas",
    "native_optimizer_step",
    "adam_m0_first_step",
    "p5_adam_m0_first_step",
    "preserve_rng_and_mode",
    "assert_training_batch_source",
    "compute_gradient_moments",
    "adam_mt_step",
    "gradients_are_finite",
    "amp_found_inf",
    "synchronized_step_decision",
    "synchronized_amp_decision",
    "advance_scaler_after_synchronized_skip",
    "compute_parameter_drift",
]
