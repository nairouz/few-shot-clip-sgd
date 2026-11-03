# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

"""Small filesystem helpers retained for the established standalone layout."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_root(path: str) -> str:
    configured = os.environ.get("DATASETS_ROOT", path)
    return str(Path(configured).expanduser().resolve())


__all__ = ["resolve_root"]
