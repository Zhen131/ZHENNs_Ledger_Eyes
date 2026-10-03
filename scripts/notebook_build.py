"""Build notebook/ledger_eyes.ipynb from the percent-format source notebook/ledger_eyes.py.

The .py file is the only file anyone edits. The .ipynb is generated from it,
always the same way: no outputs, cell ids derived from the cell content (so
regenerating an unchanged source gives a byte-identical file), and a plain
Python 3 kernel so Colab and papermill can run it.

Usage: scripts/build_notebook.sh   (or: .venv/bin/python scripts/notebook_build.py)
"""

import hashlib
import sys
from pathlib import Path

import jupytext
import nbformat

REPO_DIR = Path(__file__).resolve().parent.parent
SOURCE = REPO_DIR / "notebook" / "ledger_eyes.py"
NOTEBOOK = REPO_DIR / "notebook" / "ledger_eyes.ipynb"

NOTEBOOK_METADATA = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}


def build_notebook_text(source: Path = SOURCE) -> str:
    """Return the .ipynb JSON text that the source file should produce."""
    notebook = jupytext.read(source, fmt="py:percent")
    seen = {}
    for cell in notebook.cells:
        # Keep only the tags (for example "parameters"); drop any other cell metadata.
        tags = cell.metadata.get("tags")
        cell.metadata = {"tags": list(tags)} if tags else {}
        # Content-based id: unchanged cells keep their id when other cells change.
        base = hashlib.sha1(f"{cell.cell_type}\n{cell.source}".encode("utf-8")).hexdigest()[:8]
        seen[base] = seen.get(base, 0) + 1
        cell["id"] = base if seen[base] == 1 else f"{base}-{seen[base]}"
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    notebook.metadata = dict(NOTEBOOK_METADATA)
    notebook.nbformat, notebook.nbformat_minor = 4, 5
    nbformat.validate(notebook)
    return nbformat.writes(notebook) + "\n"


def main() -> int:
    NOTEBOOK.write_text(build_notebook_text(), encoding="utf-8")
    print(f"Wrote {NOTEBOOK.relative_to(REPO_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
