#!/usr/bin/env python3
# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

"""Parse Python files and check that documented experiment entry points exist."""
import ast
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
errors = []
files = sorted(p for p in ROOT.rglob("*.py") if ".venv" not in p.parts)
for path in files:
    try:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path.relative_to(ROOT)))
    except (SyntaxError, UnicodeError) as error:
        errors.append(f"{path.relative_to(ROOT)}: {error}")

sys.path.insert(0, str(ROOT / "scripts"))
from run_experiment import METHODS
for method, folder in METHODS.items():
    if not (ROOT / folder / "main.py").is_file():
        errors.append(f"Missing entry point for {method}")

print(f"Parsed {len(files)} Python files; checked {len(METHODS)} benchmark entry points.")
for error in errors:
    print(error)
print("This check does not import dependencies or reproduce training results.")
raise SystemExit(bool(errors))
