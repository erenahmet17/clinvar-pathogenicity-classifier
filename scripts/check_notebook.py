"""Execute all notebook code cells offline in a clean namespace from the repo root."""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    os.chdir(ROOT)
    notebook = json.loads((ROOT / "clinvar.ipynb").read_text(encoding="utf-8"))
    namespace = {"__name__": "__notebook_smoke__"}
    count = 0
    for i, cell in enumerate(notebook["cells"], 1):
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), f"notebook_cell_{i}", "exec"), namespace)
            count += 1
    print(f"Offline notebook smoke passed: {count} code cells.")


if __name__ == "__main__":
    main()
