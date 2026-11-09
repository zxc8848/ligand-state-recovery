"""Build the versioned Zenodo data archive for BondNet 2.0.

The archive intentionally contains data only.  It does not contain source code,
model checkpoints, paper sources, or results.  See the package README for the
upstream-licence review required before public deposition.
"""

from __future__ import annotations

import csv
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE_DIR = ROOT / "release"
PACKAGE_NAME = "BondNet2_data_v1.0.0"
PACKAGE_DIR = RELEASE_DIR / "zenodo_data_v1.0.0"
ARCHIVE_PATH = RELEASE_DIR / f"{PACKAGE_NAME}.zip"

PAYLOADS = (
    (
        "ccd_heavy_leakagefree.sdf",
        "CCD-derived heavy-atom ideal-coordinate control set",
        "CCD-derived; retain source attribution",
    ),
    (
        "ccd_heavy_leakagefree_manifest.csv",
        "Control-set provenance and leakage manifest",
        "CCD-derived; retain source attribution",
    ),
    (
        "pdbbind_ccd_pose500.sdf",
        "PDBBind/CCD aligned ligand-pose evaluation input",
        "PDBBind-derived; public redistribution requires terms review",
    ),
    (
        "pdbbind_ccd_pose500_manifest.csv",
        "Per-pose provenance and exclusion manifest",
        "PDBBind-derived; public redistribution requires terms review",
    ),
    (
        "diverse80k_heavy_c25_charge_h.pt",
        "Cached Diverse-80k representation used by the training protocol",
        "Derived cache; public redistribution requires source-license review",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_payloads() -> list[dict[str, str]]:
    destination = PACKAGE_DIR / "data"
    destination.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []
    for filename, purpose, provenance in PAYLOADS:
        source = ROOT / "data" / filename
        if not source.is_file():
            raise FileNotFoundError(f"Required payload is missing: {source}")
        target = destination / filename
        shutil.copy2(source, target)
        records.append(
            {
                "relative_path": f"data/{filename}",
                "bytes": str(target.stat().st_size),
                "sha256": sha256(target),
                "purpose": purpose,
                "provenance_and_release_gate": provenance,
            }
        )
    return records


def write_manifest(records: list[dict[str, str]]) -> None:
    manifest = PACKAGE_DIR / "DATA_MANIFEST.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    checksums = PACKAGE_DIR / "SHA256SUMS.txt"
    with checksums.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(f"{record['sha256']}  {record['relative_path']}\n")


def make_archive() -> None:
    if ARCHIVE_PATH.exists():
        ARCHIVE_PATH.unlink()
    with zipfile.ZipFile(
        ARCHIVE_PATH,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        for path in sorted(PACKAGE_DIR.rglob("*")):
            if path.is_file():
                archive.write(
                    path,
                    (Path(PACKAGE_NAME) / path.relative_to(PACKAGE_DIR)).as_posix(),
                )
    with zipfile.ZipFile(ARCHIVE_PATH) as archive:
        bad_file = archive.testzip()
        if bad_file:
            raise RuntimeError(f"ZIP integrity check failed for {bad_file}")


def main() -> None:
    if not PACKAGE_DIR.is_dir():
        raise FileNotFoundError(f"Package metadata directory is missing: {PACKAGE_DIR}")
    records = copy_payloads()
    write_manifest(records)
    make_archive()
    print(f"Created: {ARCHIVE_PATH}")
    print(f"Archive bytes: {ARCHIVE_PATH.stat().st_size}")
    print(f"Payload bytes: {sum(int(row['bytes']) for row in records)}")
    print("Files:")
    for record in records:
        print(f"  {record['relative_path']}  {record['bytes']} bytes")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
