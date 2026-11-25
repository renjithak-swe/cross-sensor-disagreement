"""Command line: ``python -m cross_sensor run`` for the real experiment, ``demo`` for the offline synthetic run."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from .config import load_config, parse_config
from .errors import CrossSensorError
from .logging_setup import configure_logging
from .provenance.experiment_manifest import load_manifest, now_utc

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "experiment.toml"
CACHE_ENVIRONMENT_VARIABLE = "CROSS_SENSOR_CACHE_DIR"
LOGGER = logging.getLogger("cross_sensor")


def _parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m cross_sensor")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run the experiment against the live STAC catalogue")
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run.add_argument("--out", type=Path, default=Path("results"))
    run.add_argument("--from-manifest", type=Path, help="rerun with the configuration and source items recorded in a manifest")
    run.add_argument("--no-figures", action="store_true")

    demo = commands.add_parser("demo", help="offline end-to-end run on synthetic rasters")
    demo.add_argument("--out", type=Path, default=Path("demo-output"))

    figures = commands.add_parser("figures", help="redraw figures from an existing run's outputs")
    figures.add_argument("--out", type=Path, default=Path("results"))
    return parser.parse_args(argv)


def _cache_directory() -> Path | None:
    value = os.environ.get(CACHE_ENVIRONMENT_VARIABLE)
    return Path(value) if value else None


def main(argv: list[str] | None = None) -> int:
    arguments = _parse_arguments(sys.argv[1:] if argv is None else argv)
    configure_logging(run_id=now_utc().strftime("%Y%m%dT%H%M%SZ"))
    try:
        if arguments.command == "figures":
            from .runner import rerender_figures

            print(f"figures redrawn; manifest updated at {rerender_figures(arguments.out)}")
            return 0
        if arguments.command == "demo":
            from .synthetic import run_synthetic_demo

            manifest = run_synthetic_demo(arguments.out)
        else:
            from .discovery.stac_client import PystacCatalog
            from .retrieval.raster_source import SignedAssetSource, planetary_computer_signer
            from .runner import execute, pinned_ids_from_manifest

            pinned = None
            if arguments.from_manifest:
                recorded = load_manifest(arguments.from_manifest)
                config = parse_config(recorded["config"], arguments.config.resolve().parent)
                pinned = pinned_ids_from_manifest(recorded)
            else:
                config = load_config(arguments.config)
            catalog = PystacCatalog(config.catalog.endpoint)
            source = SignedAssetSource(planetary_computer_signer, _cache_directory())
            manifest = execute(config, catalog, source, arguments.out, pinned, should_render_figures=not arguments.no_figures)
    except CrossSensorError as error:
        LOGGER.error("run failed", extra={"error_type": type(error).__name__, "detail": str(error)})
        return 1
    print(f"{manifest.experiment_id}: {len(manifest.source_observations)} pairs compared; outputs in {arguments.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
