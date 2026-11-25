"""Exception types raised by the pipeline."""


class CrossSensorError(Exception):
    """Base class for all errors raised by this package."""


class ConfigError(CrossSensorError):
    """The experiment configuration is missing a value or holds an invalid one."""


class MetadataError(CrossSensorError):
    """A STAC item or product metadata file lacks a field the pipeline needs, or holds an invalid value."""


class GridAlignmentError(CrossSensorError):
    """A raster does not sit on the lattice the comparison grid requires, or does not cover it."""


class CatalogError(CrossSensorError):
    """The STAC catalogue could not be searched."""


class RasterAccessError(CrossSensorError):
    """A raster or metadata asset could not be read."""


class UnsupportedPlatformError(CrossSensorError):
    """No documented coefficients exist for the requested platform."""
