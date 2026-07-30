"""Strict configuration loading for the European neural baseline."""

from __future__ import annotations

import hashlib
import math
import tomllib
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Final

CONFIG_SCHEMA_VERSION: Final = "european-neural-baseline-config/v1"
ARTIFACT_SCHEMA_VERSION: Final = "european-neural-artifact/v1"
MAX_HIDDEN_DIMENSION: Final = 4096
MAX_MODEL_PARAMETERS: Final = 10_000_000
FEATURE_ORDER: Final = (
    "option_type",
    "spot",
    "strike",
    "maturity",
    "rate",
    "dividend_yield",
    "volatility",
)
RAW_PHYSICAL_REPRESENTATION: Final = "raw_physical_v1"
FORWARD_NORMALIZED_REPRESENTATION: Final = "forward_normalized_v1"
REPRESENTATION_FEATURE_ORDER: Final = {
    RAW_PHYSICAL_REPRESENTATION: FEATURE_ORDER,
    FORWARD_NORMALIZED_REPRESENTATION: (
        "option_type",
        "log_forward_moneyness",
        "total_volatility",
    ),
}

_TOP_LEVEL_KEYS: Final = {
    "schema_version",
    "experiment_name",
    "representation",
    "objective",
    "features",
    "dtype",
    "device",
    "model",
    "optimizer",
    "training",
    "artifact",
    "evaluation",
}
_REQUIRED_TOP_LEVEL_KEYS: Final = _TOP_LEVEL_KEYS - {"objective", "representation"}
_SECTION_KEYS: Final = {
    "model": {"hidden_dimensions", "activation", "output_dimension"},
    "optimizer": {"name", "learning_rate", "weight_decay"},
    "training": {
        "seed",
        "batch_size",
        "max_epochs",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "num_threads",
    },
    "artifact": {"schema_version"},
    "evaluation": {
        "batch_size",
        "core_max_absolute_z",
        "tail_max_absolute_z",
    },
    "objective": {
        "name",
        "price_weight",
        "gradient_weight",
    },
}


class TrainingConfigError(ValueError):
    """Raised when a training configuration does not satisfy its contract."""


@dataclass(frozen=True)
class ModelConfig:
    hidden_dimensions: tuple[int, ...]
    activation: str
    output_dimension: int


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    learning_rate: float
    weight_decay: float


@dataclass(frozen=True)
class FitConfig:
    seed: int
    batch_size: int
    max_epochs: int
    early_stopping_patience: int
    early_stopping_min_delta: float
    num_threads: int


@dataclass(frozen=True)
class EvaluationConfig:
    batch_size: int
    core_max_absolute_z: float
    tail_max_absolute_z: float


@dataclass(frozen=True)
class ObjectiveConfig:
    name: str
    price_weight: float
    gradient_weight: float

    @property
    def uses_first_derivatives(self) -> bool:
        return self.name == "price_and_first_derivatives_v1"


@dataclass(frozen=True)
class NeuralBaselineConfig:
    experiment_name: str
    representation: str
    features: tuple[str, ...]
    dtype: str
    device: str
    model: ModelConfig
    optimizer: OptimizerConfig
    objective: ObjectiveConfig
    training: FitConfig
    artifact_schema_version: str
    evaluation: EvaluationConfig
    sha256: str


def _fail(message: str) -> TrainingConfigError:
    return TrainingConfigError(message)


def _mapping(table: dict[str, Any], key: str) -> dict[str, Any]:
    value = table.get(key)
    if not isinstance(value, dict):
        raise _fail(f"configuration section '{key}' must be a table")
    unknown = sorted(set(value) - _SECTION_KEYS[key])
    if unknown:
        raise _fail(f"configuration section '{key}' has unknown key(s): {', '.join(unknown)}")
    missing = sorted(_SECTION_KEYS[key] - set(value))
    if missing:
        raise _fail(f"configuration section '{key}' is missing key(s): {', '.join(missing)}")
    return value


def _integer(table: dict[str, Any], key: str, where: str, *, minimum: int = 1) -> int:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _fail(f"{where}.{key} must be an integer >= {minimum}, got {value!r}")
    return value


def _number(
    table: dict[str, Any],
    key: str,
    where: str,
    *,
    minimum: float,
    inclusive: bool = False,
) -> float:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _fail(f"{where}.{key} must be a number, got {value!r}")
    result = float(value)
    valid = math.isfinite(result) and (result >= minimum if inclusive else result > minimum)
    if not valid:
        operator = ">=" if inclusive else ">"
        raise _fail(f"{where}.{key} must be finite and {operator} {minimum}, got {value!r}")
    return result


def load_training_config(path: Path) -> NeuralBaselineConfig:
    """Load and validate a pinned neural-baseline TOML file."""
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise _fail(f"cannot read training configuration '{path}': {error}") from error
    try:
        table = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise _fail(f"training configuration '{path}' is not valid TOML: {error}") from error
    if not isinstance(table, dict):
        raise _fail("training configuration must be a TOML table")

    unknown = sorted(set(table) - _TOP_LEVEL_KEYS)
    if unknown:
        raise _fail(f"training configuration has unknown key(s): {', '.join(unknown)}")
    missing = sorted(_REQUIRED_TOP_LEVEL_KEYS - set(table))
    if missing:
        raise _fail(f"training configuration is missing key(s): {', '.join(missing)}")
    if table["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise _fail(
            f"schema_version must be '{CONFIG_SCHEMA_VERSION}', got {table['schema_version']!r}"
        )
    if not isinstance(table["experiment_name"], str) or not table["experiment_name"]:
        raise _fail("experiment_name must be a non-empty string")
    representation = table.get("representation", RAW_PHYSICAL_REPRESENTATION)
    if (
        not isinstance(representation, str)
        or representation not in REPRESENTATION_FEATURE_ORDER
    ):
        raise _fail(
            "representation must be one of "
            f"{sorted(REPRESENTATION_FEATURE_ORDER)}, got {representation!r}"
        )
    if table["features"] != list(FEATURE_ORDER):
        raise _fail(f"features must be exactly {list(FEATURE_ORDER)}, got {table['features']!r}")
    if table["dtype"] != "float64":
        raise _fail("dtype must be 'float64'; price and Greek validation is double precision")
    if table["device"] != "cpu":
        raise _fail("device must be 'cpu' in the deterministic stage-1 baseline")

    objective_table = table.get("objective")
    if objective_table is None:
        objective = ObjectiveConfig(
            name="price_only_v1",
            price_weight=1.0,
            gradient_weight=0.0,
        )
    else:
        objective_values = _mapping(table, "objective")
        if objective_values["name"] != "price_and_first_derivatives_v1":
            raise _fail(
                "objective.name must be 'price_and_first_derivatives_v1' "
                "when an objective table is present"
            )
        if representation != FORWARD_NORMALIZED_REPRESENTATION:
            raise _fail(
                "the first-derivative objective requires "
                f"representation = '{FORWARD_NORMALIZED_REPRESENTATION}'"
            )
        objective = ObjectiveConfig(
            name="price_and_first_derivatives_v1",
            price_weight=_number(
                objective_values,
                "price_weight",
                "objective",
                minimum=0.0,
            ),
            gradient_weight=_number(
                objective_values,
                "gradient_weight",
                "objective",
                minimum=0.0,
            ),
        )

    model = _mapping(table, "model")
    dimensions = model["hidden_dimensions"]
    if (
        not isinstance(dimensions, list)
        or not dimensions
        or any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or not 0 < item <= MAX_HIDDEN_DIMENSION
            for item in dimensions
        )
    ):
        raise _fail(
            "model.hidden_dimensions must be a non-empty list of positive "
            f"integers no larger than {MAX_HIDDEN_DIMENSION}"
        )
    layer_dimensions = (len(FEATURE_ORDER), *dimensions, 1)
    parameters = sum(
        input_size * output_size + output_size
        for input_size, output_size in pairwise(layer_dimensions)
    )
    if parameters > MAX_MODEL_PARAMETERS:
        raise _fail(
            f"model has {parameters} parameters; maximum is {MAX_MODEL_PARAMETERS}"
        )
    if model["activation"] != "tanh":
        raise _fail("model.activation must be 'tanh' for C++ SmoothMlp compatibility")
    if model["output_dimension"] != 1:
        raise _fail("model.output_dimension must be 1")

    optimizer = _mapping(table, "optimizer")
    if optimizer["name"] != "adamw":
        raise _fail("optimizer.name must be 'adamw'")

    training = _mapping(table, "training")
    seed = _integer(training, "seed", "training", minimum=0)

    artifact = _mapping(table, "artifact")
    if artifact["schema_version"] != ARTIFACT_SCHEMA_VERSION:
        raise _fail(
            f"artifact.schema_version must be '{ARTIFACT_SCHEMA_VERSION}', "
            f"got {artifact['schema_version']!r}"
        )

    evaluation = _mapping(table, "evaluation")
    core_z = _number(evaluation, "core_max_absolute_z", "evaluation", minimum=0.0)
    tail_z = _number(evaluation, "tail_max_absolute_z", "evaluation", minimum=core_z)

    return NeuralBaselineConfig(
        experiment_name=table["experiment_name"],
        representation=representation,
        features=FEATURE_ORDER,
        dtype="float64",
        device="cpu",
        model=ModelConfig(tuple(dimensions), "tanh", 1),
        optimizer=OptimizerConfig(
            name="adamw",
            learning_rate=_number(
                optimizer, "learning_rate", "optimizer", minimum=0.0
            ),
            weight_decay=_number(
                optimizer,
                "weight_decay",
                "optimizer",
                minimum=0.0,
                inclusive=True,
            ),
        ),
        objective=objective,
        training=FitConfig(
            seed=seed,
            batch_size=_integer(training, "batch_size", "training"),
            max_epochs=_integer(training, "max_epochs", "training"),
            early_stopping_patience=_integer(
                training, "early_stopping_patience", "training"
            ),
            early_stopping_min_delta=_number(
                training,
                "early_stopping_min_delta",
                "training",
                minimum=0.0,
                inclusive=True,
            ),
            num_threads=_integer(training, "num_threads", "training"),
        ),
        artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
        evaluation=EvaluationConfig(
            batch_size=_integer(evaluation, "batch_size", "evaluation"),
            core_max_absolute_z=core_z,
            tail_max_absolute_z=tail_z,
        ),
        sha256=hashlib.sha256(raw).hexdigest(),
    )
