"""
Module YamlParser
"""

from pathlib import Path
from collections.abc import Callable
from typing import Any, cast

import strictyaml  # pyright: ignore[reportMissingTypeStubs]
import yaml

# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
# )


class YamlParserError(Exception):
    """Raised when a YAML document cannot be read or parsed."""


class YamlParser:
    """
    Class YamlParser
    """

    def __init__(self) -> None:
        self.data: dict[str, Any] = {}

    def parse(self, path: str | Path) -> dict[str, Any]:
        """Parse one YAML mapping, replacing data from any previous parse."""
        source = Path(path)
        try:
            yaml_text = source.read_text(encoding="utf-8")
        except OSError as err:
            raise YamlParserError(f"cannot read {source}: {err}") from err
        try:
            # StrictYAML rejects duplicate keys and unsafe YAML features. PyYAML
            # is then used only to retain scalar types for strict model validation.
            strict_load = cast(
                Callable[[str], object],
                getattr(strictyaml, "load"),
            )
            strict_load(yaml_text)
            yaml_data = cast(object, yaml.safe_load(yaml_text))
        except strictyaml.YAMLError as err:
            raise YamlParserError(f"invalid YAML in {source}: {err}") from err
        except yaml.YAMLError as err:
            raise YamlParserError(f"invalid YAML in {source}: {err}") from err
        if not isinstance(yaml_data, dict):
            raise YamlParserError(f"{source} must contain a YAML mapping at the root")
        data = cast(dict[object, object], yaml_data)
        if not all(isinstance(key, str) for key in data):
            raise YamlParserError(f"{source} mapping keys must all be strings")
        self.data = cast(dict[str, Any], dict(data))
        return self.data
