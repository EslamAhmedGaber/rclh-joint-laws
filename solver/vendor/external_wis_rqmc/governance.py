"""Fail-closed manifest and artifact helpers for the external comparator."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .cases import FROZEN_CASES, FROZEN_HURST_VALUES


class GovernanceError(RuntimeError):
    """Raised when a prospective freeze invariant is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise GovernanceError("manifest root must be an object")
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Validate frozen scientific and sampling semantics."""

    if manifest.get("protocol") != "external_wis_rqmc_benchmark_v1":
        raise GovernanceError("unexpected protocol")
    if manifest.get("evidence_state") != "PROSPECTIVE_UNRUN":
        raise GovernanceError("manifest is not prospectively unrun")
    if tuple(manifest.get("hurst_values", ())) != FROZEN_HURST_VALUES:
        raise GovernanceError("Hurst grid differs from the frozen four values")
    cases = manifest.get("cases")
    if not isinstance(cases, dict) or tuple(cases) != tuple(FROZEN_CASES):
        raise GovernanceError("case order or membership is not G/R/C/L")
    for code, frozen in FROZEN_CASES.items():
        observed = cases[code]
        expected = {
            "alpha": frozen.alpha,
            "beta": frozen.beta,
            "initial_value": frozen.initial_value,
            "drift": frozen.drift_name,
        }
        if observed != expected:
            raise GovernanceError(f"case {code} differs from its freeze")
    reporting_nodes = np.asarray(manifest.get("reporting_nodes"), dtype=float)
    expected_nodes = np.linspace(0.0, 1.0, 33)
    if not np.array_equal(reporting_nodes, expected_nodes):
        raise GovernanceError("primary reporting nodes are not the fixed 33 nodes")
    if manifest.get("output_density_node_counts") != [17, 33, 65, 129]:
        raise GovernanceError("output-density lane differs from the freeze")
    sampling = manifest.get("sampling", {})
    seeds = sampling.get("scramble_seeds", ())
    if len(seeds) != 16 or len(set(seeds)) != 16:
        raise GovernanceError("exactly 16 distinct scramble seeds are required")
    if sampling.get("scrambles") != 16:
        raise GovernanceError("scramble count must be 16")
    if sampling.get("familywise_confidence") != 0.99:
        raise GovernanceError("familywise confidence must be 0.99")
    expected_family_size = 4 * 4 * 3 * 33
    if sampling.get("global_family_size") != expected_family_size:
        raise GovernanceError("global family size must cover all H/case/q/time cells")
    factor_levels = manifest.get("factor", {}).get("levels", ())
    if (
        len(factor_levels) < 2
        or sorted(set(factor_levels)) != list(factor_levels)
        or any(int(level) < 1 for level in factor_levels)
    ):
        raise GovernanceError("factor levels must be an increasing refinement ladder")
    exponents = sampling.get("fine_exponent_ladder", ())
    if (
        len(exponents) < 1
        or sorted(set(exponents)) != list(exponents)
        or any(int(value) < 2 for value in exponents)
    ):
        raise GovernanceError("RQMC exponents must be an increasing ladder")
    if manifest.get("claims", {}).get("novelty") != "PROVISIONAL":
        raise GovernanceError("novelty must remain provisional")


def validate_source_hashes(
    manifest: dict[str, Any],
    repository_root: Path,
) -> dict[str, str]:
    """Recompute every prospectively frozen component hash."""

    expected = manifest.get("source_hashes")
    if not isinstance(expected, dict) or not expected:
        raise GovernanceError("source_hashes are missing")
    root = Path(repository_root).resolve()
    observed: dict[str, str] = {}
    for relative, expected_hash in expected.items():
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise GovernanceError("hashed path escapes repository root") from error
        if not path.is_file():
            raise GovernanceError(f"hashed source is missing: {relative}")
        digest = sha256_file(path)
        observed[relative] = digest
        if digest != expected_hash:
            raise GovernanceError(f"source hash mismatch: {relative}")
    return observed


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON artifact atomically without replacing other evidence."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


__all__ = [
    "GovernanceError",
    "atomic_write_json",
    "load_manifest",
    "sha256_file",
    "validate_manifest",
    "validate_source_hashes",
]
