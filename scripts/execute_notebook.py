"""Verify restart-and-run-all in an isolated Jupyter kernel using this interpreter."""

import json
import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = ROOT / "validation_logs" / "kernels" / "clinvar-verify"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "ClinVar isolated verification",
                "language": "python",
                "env": {"IPYTHONDIR": str(ROOT / "validation_logs" / "ipython")},
            }
        ),
        encoding="utf-8",
    )
    specs = KernelSpecManager(kernel_dirs=[str(directory.parent)])
    manager = KernelManager(kernel_name="clinvar-verify", kernel_spec_manager=specs)
    notebook = nbformat.read(ROOT / "clinvar.ipynb", as_version=4)
    client = NotebookClient(
        notebook, km=manager, timeout=60, resources={"metadata": {"path": str(ROOT)}}
    )
    try:
        client.execute()
        nbformat.write(notebook, ROOT / "validation_logs" / "executed_notebook.ipynb")
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    print("Isolated Jupyter restart-and-run-all passed.")


if __name__ == "__main__":
    main()
