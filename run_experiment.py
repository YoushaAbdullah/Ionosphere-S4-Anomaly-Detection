"""Run the research notebook unchanged outside Colab and capture real outputs.

The only adapters replace Colab's interactive file picker/download and capture
display/plot output. All scientific code comes from the source notebook. No
models, labels, thresholds, features, or splits are reimplemented here.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import csv
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
import types
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "Q4 Paper Code" / "scintillation_anomaly_detection.ipynb"
DATA_SHA256 = "e7b2e4882ac7c856d438b4e2210993c1a643af52abb6f59ca19c79afe175504e"
PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "deap",
            "ipython", "Jinja2", "MarkupSafe", "joblib", "threadpoolctl")
JARS = {
    "weka-stable-3.8.6.jar": (
        "https://repo.maven.apache.org/maven2/nz/ac/waikato/cms/weka/weka-stable/3.8.6/weka-stable-3.8.6.jar",
        "932ea2f342b58fe45736389e9c426d5b5955e610d1f5f6485117f532068915e9"),
    "bounce-0.18.jar": (
        "https://repo.maven.apache.org/maven2/nz/ac/waikato/cms/weka/thirdparty/bounce/0.18/bounce-0.18.jar",
        "bffff1505335c02256b7ab2ccffbe4aa4d3ac9fe14c17557809b7c9d99d666ca"),
}
REQUIRED_COLUMNS = {
    "YYYY", "MM", "DD", "Second", "PRN", "Azimuth", "Elevation", "L1 CNo",
    "Total S4", "Correction to total S4", "Average of Carrier divergence",
    "Sigma of Carrier Divergence", "TEC at TOW", "L1 Lock time", "L2 CNo",
}


def sha256(path):
    """Hash in chunks so checking the archive does not duplicate it in RAM."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_inputs(data, allow_different=False):
    """Reject the wrong archive before starting expensive fitting."""
    digest = sha256(data)
    if digest != DATA_SHA256 and not allow_different:
        raise ValueError("Dataset SHA-256 differs from the recorded 2019 archive. "
                         "Use the original data, or explicitly pass --allow-different-data "
                         "for a new experiment; its results are not the 2019 reproduction.")
    with zipfile.ZipFile(data) as archive:
        names = sorted(name for name in archive.namelist()
                       if name.lower().endswith(".csv") and "__MACOSX" not in name
                       and not name.rsplit("/", 1)[-1].startswith("._"))
        if not names:
            raise ValueError("No usable CSV files in archive.")
        with archive.open(names[0]) as stream:
            text = io.TextIOWrapper(stream, encoding="utf-8-sig")
            next(text)
            next(text)
            columns = {value.strip() for value in next(csv.reader(text))[:19]}
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"First CSV is missing required fields: {sorted(missing)}")
    return {"sha256": digest, "matches_recorded_2019": digest == DATA_SHA256,
            "bytes": data.stat().st_size, "csv_files": len(names)}


def check_runtime():
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Use Python 3.12 for this pinned reproduction environment.")
    versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    # Check every resolved dependency, not just the direct scientific packages.
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        specification = line.strip().rstrip("\\").strip()
        if "==" not in specification or specification.startswith("#"):
            continue
        name, expected = specification.split("==", 1)
        expected = expected.split(";", 1)[0].strip()
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise RuntimeError(f"{name}=={actual} is installed; expected {expected}. "
                               "Install requirements.txt in a fresh virtual environment.")
    if not shutil.which("java"):
        raise RuntimeError("Install a JDK (recommended: OpenJDK/Temurin 21) and put java on PATH.")
    java = subprocess.run(["java", "-version"], capture_output=True, text=True, check=True)
    compiler = subprocess.run(["java", "--list-modules"], capture_output=True, text=True, check=True)
    if "jdk.compiler@" not in compiler.stdout:
        raise RuntimeError("A full JDK is required, not a Java-only runtime (JRE).")
    return versions, (java.stderr or java.stdout).strip()


class Tee(io.TextIOBase):
    """Mirror actual cell text to the terminal, run log, and notebook output."""
    def __init__(self, terminal, log, captured):
        self.terminal, self.log, self.captured = terminal, log, captured

    def write(self, text):
        for stream in (self.terminal, self.log, self.captured):
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in (self.terminal, self.log, self.captured):
            stream.flush()


def run(args):
    data = args.data.expanduser().resolve()
    input_info = check_inputs(data, args.allow_different_data)
    versions, java = check_runtime()
    print(json.dumps({"input": input_info, "versions": versions, "java": java}, indent=2))
    if args.check_only:
        print("Preflight passed; no experiment or network dependency downloads were run.")
        return
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty. Choose a new --output to preserve prior results.")
    output.mkdir(parents=True, exist_ok=True)
    source_bytes = NOTEBOOK.read_bytes()
    notebook = json.loads(source_bytes)
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell.update(outputs=[], execution_count=None)
    manifest = {
        "status": "running", "source": "Q4 Paper Code/scintillation_anomaly_detection.ipynb",
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "input": input_info, "python": sys.version, "platform": platform.platform(),
        "versions": versions, "java": java,
        "adapters": ["Colab upload/download", "display and PNG capture"],
        "cv_cache": "disabled; fresh folds", "cells": [],
    }
    os.environ["Q4_PAPER_OUTPUT"] = str(output / "study_iii_results")
    os.environ["Q4_INPUT_SHA256"] = input_info["sha256"]
    os.environ["Q4_PREPARATION_SHA256"] = hashlib.sha256(json.dumps(
        [cell["source"] for cell in notebook["cells"][:12]], ensure_ascii=False).encode()).hexdigest()
    os.environ.pop("Q4_CV_CACHE", None)
    os.environ["MPLBACKEND"] = "Agg"
    if not args.prepare_only:
        dependencies = output / "study_iii_results" / "dependencies"
        dependencies.mkdir(parents=True)
        for filename, (url, expected) in JARS.items():
            target = dependencies / filename
            with urllib.request.urlopen(url, timeout=60) as response, target.open("wb") as destination:
                shutil.copyfileobj(response, destination)
            if sha256(target) != expected:
                raise ValueError(f"Dependency checksum mismatch: {filename}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from threadpoolctl import threadpool_info
    manifest["threadpools"] = threadpool_info()

    # Install only the two Colab file operations used by the original notebook.
    files = types.ModuleType("google.colab.files")
    files.upload = lambda: {data.name: data.read_bytes()}
    files.download = lambda path: print(f"Artifact retained: {Path(path).name}")
    colab = types.ModuleType("google.colab")
    colab.files = files
    try:
        import google
    except ImportError:
        google = types.ModuleType("google")
        google.__path__ = []
        sys.modules["google"] = google
    google.colab = colab
    sys.modules["google.colab"] = colab
    sys.modules["google.colab.files"] = files

    outputs = []
    counter = 0

    def display(*objects, **kwargs):
        for obj in objects:
            content = {"text/plain": repr(obj)}
            html = getattr(obj, "_repr_html_", None)
            if html:
                rendered = html()
                if rendered:
                    content["text/html"] = rendered
            outputs.append({"output_type": "display_data", "metadata": {}, "data": content})

    def show(*unused_args, **unused_kwargs):
        for number in plt.get_fignums():
            figure = plt.figure(number)
            stream = io.BytesIO()
            figure.savefig(stream, format="png")
            outputs.append({"output_type": "display_data", "metadata": {},
                            "data": {"image/png": base64.b64encode(stream.getvalue()).decode("ascii")}})
            plt.close(figure)

    plt.show = show
    namespace = {"__name__": "__main__", "display": display}
    notebook["metadata"]["kernelspec"] = {"display_name": "Python 3.12", "language": "python", "name": "python3"}
    notebook["metadata"]["language_info"] = {"name": "python", "version": platform.python_version()}

    def save():
        notebook["metadata"]["q4_execution"] = manifest
        (output / "execution_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (output / "2019_study_iii_executed.ipynb").write_text(json.dumps(notebook, indent=1), encoding="utf-8")

    with (output / "execution.log").open("w", encoding="utf-8") as log:
        try:
            for number, cell in enumerate(notebook["cells"], 1):
                if args.prepare_only and number > 12:
                    break
                if cell["cell_type"] != "code" or not "".join(cell["source"]).strip():
                    continue
                counter += 1
                outputs = []
                captured = io.StringIO()
                cell.update(execution_count=counter, outputs=outputs)
                entry = {"cell": number, "status": "running"}
                manifest["cells"].append(entry)
                print(f"Running notebook cell {number} …", flush=True)
                started = time.monotonic()
                with contextlib.redirect_stdout(Tee(sys.stdout, log, captured)), \
                     contextlib.redirect_stderr(Tee(sys.stderr, log, captured)):
                    try:
                        exec(compile("".join(cell["source"]), f"research-notebook:cell-{number}", "exec"), namespace)
                    except BaseException as error:
                        entry.update(status="failed", error=str(error))
                        outputs.append({"output_type": "error", "ename": type(error).__name__,
                                        "evalue": str(error), "traceback": traceback.format_exc().splitlines()})
                        raise
                    finally:
                        outputs.insert(0, {"output_type": "stream", "name": "stdout", "text": captured.getvalue()})
                        entry["seconds"] = time.monotonic() - started
                entry["status"] = "completed"
                save()
            manifest["status"] = "prepared" if args.prepare_only else "completed"
        except BaseException:
            manifest["status"] = "failed"
            raise
        finally:
            for name in ("df", "df_clean", "train_df", "val_df", "test_df"):
                frame = namespace.get(name)
                if frame is not None:
                    info = {"rows": len(frame)}
                    if "LABEL" in frame:
                        info["label_counts"] = {str(k): int(v) for k, v in frame.LABEL.value_counts().items()}
                    manifest[name] = info
            save()
    print(f"Finished: {output / '2019_study_iii_executed.ipynb'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Path to the original 2019.zip")
    parser.add_argument("--output", type=Path, default=Path("runs/2019"), help="New/empty output directory")
    parser.add_argument("--check-only", action="store_true", help="Validate pinned dependencies, JDK and data without fitting")
    parser.add_argument("--prepare-only", action="store_true", help="Execute unchanged preparation and EDA only")
    parser.add_argument("--allow-different-data", action="store_true", help="Explicitly start a different-data experiment")
    try:
        run(parser.parse_args())
    except (ValueError, FileNotFoundError, RuntimeError, importlib.metadata.PackageNotFoundError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
