"""KV Rosetta — portable KV-cache exchange across models, runtimes and backends.

Only the manifest layer is re-exported here. Everything that needs numpy or torch
(container, adapters, mappers) must be imported from its own module, so that
validating a manifest stays a stdlib-only operation.
"""

from kv_rosetta.manifest import (
    ACCEPTED_SCHEMAS,
    DTYPES,
    LAYOUT,
    SCHEMA,
    SCHEMA_LATEST,
    ManifestError,
    ModelABI,
    compatibility,
    load,
)

__all__ = [
    "ACCEPTED_SCHEMAS",
    "DTYPES",
    "LAYOUT",
    "SCHEMA",
    "SCHEMA_LATEST",
    "ManifestError",
    "ModelABI",
    "compatibility",
    "load",
]
