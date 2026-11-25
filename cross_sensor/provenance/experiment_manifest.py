"""Experiment manifest: what was run, on which source observations, producing which files."""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from ..errors import ConfigError
from .hashing import sha256_file, sha256_json

MANIFEST_VERSION = 1
TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "pyarrow",
    "rasterio",
    "pyproj",
    "shapely",
    "geopandas",
    "pystac-client",
    "planetary-computer",
    "matplotlib",
)


def strip_query_string(href: str) -> str:
    """Remove a query string such as a temporary SAS signature, so tokens never reach the manifest."""
    parts = urlsplit(href)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def collect_software_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed"
    try:
        import rasterio

        versions["gdal"] = rasterio.__gdal_version__
    except ImportError:
        versions["gdal"] = "not installed"
    return versions


def read_git_commit(directory: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=directory, capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = completed.stdout.strip()
    return commit if completed.returncode == 0 and commit else None


@dataclass
class OutputRecord:
    path: str
    sha256: str
    size_bytes: int


@dataclass
class ExperimentManifest:
    experiment_id: str
    executed_at: str
    catalog_endpoint: str
    collections: dict[str, str]
    aoi: dict[str, Any]
    config: dict[str, Any]
    config_sha256: str
    grid: dict[str, Any]
    resampling: dict[str, str]
    quality_masks: dict[str, Any]
    software_versions: dict[str, str]
    git_commit: str | None
    source_observations: list[dict[str, Any]]
    outputs: list[OutputRecord] = field(default_factory=list)
    manifest_version: int = MANIFEST_VERSION

    def record_outputs(self, output_directory: Path, paths: list[Path]) -> None:
        for path in sorted(set(paths)):
            self.outputs.append(
                OutputRecord(
                    path=path.relative_to(output_directory).as_posix(),
                    sha256=sha256_file(path),
                    size_bytes=path.stat().st_size,
                )
            )

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, default=str) + "\n", encoding="utf-8")


def build_experiment_id(config_raw: dict[str, Any], executed_at: datetime) -> str:
    return f"xsensor-{executed_at.strftime('%Y%m%dT%H%M%SZ')}-{sha256_json(config_raw)[:8]}"


def now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError(f"cannot read manifest {path}: {error}") from error
    if document.get("manifest_version") != MANIFEST_VERSION:
        raise ConfigError(f"unsupported manifest version in {path}")
    for key in ("config", "source_observations"):
        if key not in document:
            raise ConfigError(f"manifest {path} has no '{key}'")
    return document
