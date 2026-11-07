"""Fail-closed artifact governance for Solver Accuracy V2."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


class AccuracyGovernanceError(RuntimeError):
    """Raised when a prospective Solver Accuracy V2 invariant is violated."""


FORBIDDEN_FIELDS = frozenset(
    {
        "wall_seconds",
        "solve_seconds",
        "setup_seconds",
        "elapsed_seconds",
        "duration_seconds",
        "cpu_seconds",
        "rss",
        "rss_bytes",
        "speed",
        "speedup",
        "throughput",
        "timing",
        "performance_metrics",
        "memory_bytes",
    }
)

TARGET_MANIFEST_KEYS = frozenset(
    {
        "protocol",
        "evidence_state",
        "performance_evidence_allowed",
        "cell",
        "reporting_nodes",
        "moment_semantics",
        "refinement_axes",
        "method_allowance_construction",
        "producer_entrypoint",
        "solver_implementation_fingerprint",
        "source_closure_attested",
        "producer_controls",
        "resource_caps",
        "source_hashes",
        "claims",
    }
)


def canonical_json_bytes(payload: object) -> bytes:
    try:
        return (
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise AccuracyGovernanceError("payload is not canonical finite JSON") from error


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def load_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise AccuracyGovernanceError("JSON root must be an object")
    return payload


def require_exact_keys(
    payload: Mapping[str, object],
    expected: set[str] | frozenset[str],
    label: str,
) -> None:
    observed = set(payload)
    if observed != set(expected):
        raise AccuracyGovernanceError(
            f"{label} schema mismatch; "
            f"missing={sorted(set(expected)-observed)}, "
            f"extra={sorted(observed-set(expected))}"
        )


def reject_forbidden_fields(payload: object, trail: str = "root") -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            name = str(key)
            if name.lower() in FORBIDDEN_FIELDS:
                raise AccuracyGovernanceError(
                    f"forbidden performance field at {trail}.{name}"
                )
            reject_forbidden_fields(value, f"{trail}.{name}")
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            reject_forbidden_fields(value, f"{trail}[{index}]")


def validate_target_manifest_v2(
    manifest: Mapping[str, object],
    repository_root: Path,
    *,
    require_frozen_hashes: bool,
) -> dict[str, str]:
    """Validate the shared target schema and recompute frozen source hashes."""

    require_exact_keys(manifest, TARGET_MANIFEST_KEYS, "target manifest")
    if manifest["protocol"] != "target_wis_moment_curves_v2":
        raise AccuracyGovernanceError("target protocol changed")
    if manifest["evidence_state"] != "PROSPECTIVE_BEFORE_TARGET_EXECUTION":
        raise AccuracyGovernanceError("target manifest is not prospective")
    if manifest["performance_evidence_allowed"] is not False:
        raise AccuracyGovernanceError("target manifest permits performance evidence")
    cell = manifest["cell"]
    if (
        not isinstance(cell, Mapping)
        or set(cell) != {"case", "hurst"}
        or cell["case"] not in {"R", "C", "L1"}
        or float(cell["hurst"]) not in {0.1, 0.25, 0.5, 0.75}
    ):
        raise AccuracyGovernanceError("target cell is invalid")
    if manifest["reporting_nodes"] != [index / 32.0 for index in range(33)]:
        raise AccuracyGovernanceError("target reporting nodes changed")
    if manifest["moment_semantics"] != "coherent_full_output_v1":
        raise AccuracyGovernanceError("target moment semantics changed")
    if manifest["refinement_axes"] != [
        "factor_grid",
        "time",
        "support_chaos",
        "projector",
    ]:
        raise AccuracyGovernanceError("target refinement axes changed")
    if manifest["method_allowance_construction"] != (
        "sum_of_four_resolved_geometric_tail_indicators_plus_roundoff_v1"
    ):
        raise AccuracyGovernanceError("method allowance construction changed")
    entrypoint = manifest["producer_entrypoint"]
    if entrypoint != "05_experiments/run_solver_accuracy_v2.py":
        raise AccuracyGovernanceError("target producer entrypoint changed")
    if manifest["solver_implementation_fingerprint"] != (
        "656e9a19bec2b04a02681c233a2639db3e2e1aa1039624ee5acb2493158130ab"
    ):
        raise AccuracyGovernanceError("official fused implementation changed")
    if manifest["source_closure_attested"] is not True:
        raise AccuracyGovernanceError("source closure is not attested")
    if not isinstance(manifest["producer_controls"], Mapping) or not manifest[
        "producer_controls"
    ]:
        raise AccuracyGovernanceError("producer controls are missing")
    caps = manifest["resource_caps"]
    if (
        not isinstance(caps, Mapping)
        or not caps
        or any(type(value) is not int or value <= 0 for value in caps.values())
    ):
        raise AccuracyGovernanceError("resource caps are invalid")
    if manifest["claims"] != {
        "continuum_theorem": False,
        "rigorous_a_posteriori_bound": False,
        "novelty": "PROVISIONAL",
    }:
        raise AccuracyGovernanceError("target claim boundary changed")
    expected = manifest["source_hashes"]
    if not isinstance(expected, Mapping) or not expected or entrypoint not in expected:
        raise AccuracyGovernanceError("target source closure is incomplete")
    root = Path(repository_root).resolve()
    observed: dict[str, str] = {}
    for relative, digest in expected.items():
        if require_frozen_hashes and (not is_sha256(digest) or digest == "PENDING"):
            raise AccuracyGovernanceError("target source digest is not frozen")
        source = (root / str(relative)).resolve()
        try:
            source.relative_to(root)
        except ValueError as error:
            raise AccuracyGovernanceError("target source path escapes repository") from error
        if not source.is_file():
            raise AccuracyGovernanceError(f"target source is missing: {relative}")
        actual = sha256_file(source)
        if require_frozen_hashes and actual != digest:
            raise AccuracyGovernanceError(f"target source hash mismatch: {relative}")
        observed[str(relative)] = actual
    reject_forbidden_fields(manifest)
    return observed


def safe_attempt_id(value: str) -> str:
    selected = str(value)
    if (
        not 1 <= len(selected) <= 80
        or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for character in selected
        )
    ):
        raise AccuracyGovernanceError("attempt id contains unsafe characters")
    return selected


def exclusive_write(path: Path, payload: bytes) -> None:
    destination = Path(path)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        raise


def atomic_progress_write(path: Path, payload: Mapping[str, object]) -> None:
    """Replace only the designated non-evidence progress file atomically."""

    destination = Path(path)
    temporary = destination.with_name(f".{destination.name}.tmp")
    data = canonical_json_bytes(dict(payload))
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def recursive_member_hashes(directory: Path) -> dict[str, str]:
    root = Path(directory).resolve()
    output: dict[str, str] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == "package_manifest.json":
            continue
        output[relative] = sha256_file(path)
    if not output:
        raise AccuracyGovernanceError("package has no members")
    return output


def validate_directory_package(
    package: Path,
    *,
    protocol: str,
) -> dict[str, object]:
    directory = Path(package).resolve()
    if not directory.is_dir() or not directory.name.endswith("_package"):
        raise AccuracyGovernanceError("terminal package is missing")
    package_manifest = load_json(directory / "package_manifest.json")
    require_exact_keys(
        package_manifest,
        {"protocol", "member_sha256", "package_fingerprint", "reservation_sha256"},
        "package manifest",
    )
    if package_manifest["protocol"] != protocol:
        raise AccuracyGovernanceError("package protocol mismatch")
    observed = recursive_member_hashes(directory)
    if observed != package_manifest["member_sha256"]:
        raise AccuracyGovernanceError("package member hashes changed")
    fingerprint = sha256_bytes(canonical_json_bytes(observed))
    if package_manifest["package_fingerprint"] != fingerprint:
        raise AccuracyGovernanceError("package fingerprint changed")
    reject_forbidden_fields(package_manifest)
    return package_manifest


__all__ = [
    "AccuracyGovernanceError",
    "FORBIDDEN_FIELDS",
    "TARGET_MANIFEST_KEYS",
    "atomic_progress_write",
    "canonical_json_bytes",
    "exclusive_write",
    "is_sha256",
    "load_json",
    "recursive_member_hashes",
    "reject_forbidden_fields",
    "require_exact_keys",
    "safe_attempt_id",
    "sha256_bytes",
    "sha256_file",
    "validate_directory_package",
    "validate_target_manifest_v2",
]
