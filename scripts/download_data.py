"""Download the Kaggle/Dal Pozzolo credit card fraud dataset from Zenodo.

Usage:
    python scripts/download_data.py [--out <path>] [--force]

Skips download if the target file already exists with non-zero size.

Dataset source:
    Dal Pozzolo, A., Caelen, O., Johnson, R.A. and Bontempi, G. (2015)
    'Calibrating probability with undersampling for unbalanced classification',
    2015 IEEE SSCI, pp. 159-166.
    https://zenodo.org/records/7395559
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ZENODO_URL = "https://zenodo.org/records/7395559/files/creditcard.csv?download=1"
EXPECTED_MIN_BYTES = 140 * 1024 * 1024  # File is ~150 MB; guard against truncated downloads.
EXPECTED_ROWS = 284_807  # From Dal Pozzolo et al. (2015).


def download(url: str, out: Path) -> None:
    """Stream-download ``url`` to ``out`` with a minimal progress log.

    Streaming avoids loading the full ~150 MB file into memory, which matters
    on HPC login nodes that can have aggressive per-process memory caps.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {out}", flush=True)
    with urllib.request.urlopen(url) as response, open(out, "wb") as fh:
        total = int(response.headers.get("Content-Length") or 0)
        downloaded = 0
        chunk = 1024 * 1024
        last_print = 0
        while True:
            data = response.read(chunk)
            if not data:
                break
            fh.write(data)
            downloaded += len(data)
            # Print at most every ~5 MB to keep SLURM logs readable.
            if downloaded - last_print >= 5 * 1024 * 1024:
                pct = (100.0 * downloaded / total) if total else 0.0
                print(f"  {downloaded/1e6:7.1f} / {total/1e6:7.1f} MB ({pct:5.1f}%)", flush=True)
                last_print = downloaded
    print("Download complete.", flush=True)


def verify(path: Path) -> None:
    """Sanity-check size, header, and row count."""
    size = path.stat().st_size
    if size < EXPECTED_MIN_BYTES:
        sys.exit(f"Downloaded file is too small ({size} bytes); expected >{EXPECTED_MIN_BYTES}.")
    with open(path, "r", encoding="utf-8") as fh:
        header = fh.readline().strip()
        if "Class" not in header or "Amount" not in header:
            sys.exit(f"Unexpected header: {header!r}")
        rows = sum(1 for _ in fh)
    if rows != EXPECTED_ROWS:
        sys.exit(f"Row count {rows} != expected {EXPECTED_ROWS}")
    print(f"Verified: {size/1e6:.1f} MB, {rows} rows, header OK.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "creditcard.csv",
        help="Output path for creditcard.csv (default: repo root).",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if file exists.")
    args = parser.parse_args()

    if args.out.exists() and args.out.stat().st_size > 0 and not args.force:
        print(
            f"{args.out} already exists ({args.out.stat().st_size/1e6:.1f} MB). "
            "Use --force to re-download."
        )
    else:
        download(ZENODO_URL, args.out)
    verify(args.out)


if __name__ == "__main__":
    main()
