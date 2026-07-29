"""Explicit research stages; later stages must not bypass earlier validation."""

from enum import StrEnum


class ResearchStage(StrEnum):
    """The planned simple-to-complex pricing progression."""

    EUROPEAN_OPTION = "european-option"
    AMERICAN_OPTION = "american-option"
    EUROPEAN_SWAPTION = "european-swaption"
    BERMUDAN_SWAPTION = "bermudan-swaption"

