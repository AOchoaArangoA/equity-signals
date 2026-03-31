"""cli — command-line interface for the equity-signals package.

Entry points defined here are registered in ``pyproject.toml`` under
``[project.scripts]`` and become available as shell commands after installation.
"""

import sys

from pydantic import ValidationError
from pydantic_settings import BaseSettings

from equity_signals.config import Settings

# Fields that must be present for the pipeline to function.
_REQUIRED_FIELDS = list(Settings.model_fields.keys())


def _check_settings() -> tuple[list[str], list[str]]:
    """Attempt to load Settings and classify each field as present or missing.

    Returns:
        A two-tuple ``(present, missing)`` where each element is a list of
        field names.
    """
    try:
        settings = Settings()
        present = [
            field
            for field in _REQUIRED_FIELDS
            if getattr(settings, field, None) not in (None, "")
        ]
        missing = [f for f in _REQUIRED_FIELDS if f not in present]
        return present, missing
    except ValidationError as exc:
        missing_fields = {e["loc"][0] for e in exc.errors() if e["loc"]}
        present = [f for f in _REQUIRED_FIELDS if f not in missing_fields]
        missing = list(missing_fields)
        return present, missing


def validate_config() -> None:
    """Print which configuration keys are present and which are missing.

    Actual values are never displayed — only whether each key is set.
    Exits with status code 1 when any key is missing.
    """
    present, missing = _check_settings()

    print("equity-signals config validation")
    print("=" * 34)

    for field in _REQUIRED_FIELDS:
        status = "OK     " if field in present else "MISSING"
        print(f"  [{status}]  {field.upper()}")

    print()
    if missing:
        print(f"Result: {len(missing)} key(s) missing. Set them in your .env file.")
        sys.exit(1)
    else:
        print("Result: all keys present.")
