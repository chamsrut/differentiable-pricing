"""Derive a European-bound-constrained artifact without retraining."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from ..data.generate import sha256_file
from .artifact import (
    ARTIFACT_MANIFEST,
    WEIGHTS_FILE,
    ArtifactError,
    load_physical_model,
    write_json,
)
from .model import (
    EUROPEAN_BOUNDS_CONSTRAINT,
    EUROPEAN_BOUNDS_DIFFERENTIABILITY,
    EUROPEAN_BOUNDS_PROJECTION,
    NO_OUTPUT_CONSTRAINT,
)


class ConstraintError(RuntimeError):
    """Raised when a constrained artifact cannot be derived safely."""


def derive_bounded_artifact(source: Path, output: Path) -> dict[str, Any]:
    """Copy verified weights and attach a deterministic European projection."""
    if output.exists():
        raise ConstraintError(f"output '{output}' already exists; refusing to overwrite")
    model, source_payload = load_physical_model(source)
    if model.output_constraint != NO_OUTPUT_CONSTRAINT:
        raise ConstraintError("source artifact already has an output constraint")

    try:
        source_manifest_sha256 = sha256_file(source / ARTIFACT_MANIFEST)
    except OSError as error:
        raise ConstraintError(
            f"cannot hash source artifact manifest '{source / ARTIFACT_MANIFEST}': "
            f"{error}"
        ) from error
    weights_sha256 = source_payload["weights"]["sha256"]
    payload = copy.deepcopy(source_payload)
    payload["output_constraint"] = {
        "name": EUROPEAN_BOUNDS_CONSTRAINT,
        "projection": EUROPEAN_BOUNDS_PROJECTION,
        "bounds": (
            "discounted European call/put lower and upper bounds under the "
            "artifact's physical rate and dividend inputs"
        ),
        "differentiability": EUROPEAN_BOUNDS_DIFFERENTIABILITY,
        "source_artifact": {
            "manifest_sha256": source_manifest_sha256,
            "weights_sha256": weights_sha256,
        },
    }
    payload["limitations"] = [
        *payload["limitations"],
        (
            "European price bounds are enforced by a piecewise-differentiable "
            "projection; clipping-boundary derivatives are not unique."
        ),
    ]

    temporary: Path | None = None
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent)
        )
        shutil.copyfile(source / WEIGHTS_FILE, temporary / WEIGHTS_FILE)
        if sha256_file(temporary / WEIGHTS_FILE) != weights_sha256:
            raise ConstraintError("copied weights do not match the source digest")
        write_json(temporary / ARTIFACT_MANIFEST, payload)
        temporary.rename(output)
    except (OSError, TypeError, ValueError) as error:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
        raise ConstraintError(
            f"cannot publish constrained artifact '{output}': {error}"
        ) from error
    except ConstraintError:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
        raise
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        payload = derive_bounded_artifact(arguments.artifact, arguments.output)
    except (ArtifactError, ConstraintError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "artifact": str(arguments.output),
                "constraint": payload["output_constraint"]["name"],
                "source_manifest_sha256": payload["output_constraint"][
                    "source_artifact"
                ]["manifest_sha256"],
                "weights_sha256": payload["weights"]["sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
