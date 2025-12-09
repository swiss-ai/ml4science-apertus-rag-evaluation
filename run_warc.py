#!/usr/bin/env python3
"""
WARC extraction driver for Clariden/WSL.

What it does:
- Finds all .warc and .warc.gz files under WARC_INPUT_DIR
- Phase 1: extract HTML files to 19945_html_files/
- Phase 2: extract PDF files to 19945_pdf_files/
- Prints per-5-file progress with speed and ETA
- Streams warc_extractor.py output (shows "parsing ..." per WARC)
"""

import os
import sys
import time
import glob
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from prep_warc_files import (
    warc_to_html,
    warc_to_pdf,
    warc_to_string,
    get_base_site_from_url,
)

# Paths
WARC_INPUT_DIR = "/iopsstor/scratch/cscs/rashitig/19945/"
COLL = "19945"


def format_eta(seconds_left: float) -> str:
    if seconds_left < 0 or seconds_left is None:
        return "ETA=0s"
    hours = int(seconds_left // 3600)
    minutes = int((seconds_left % 3600) // 60)
    seconds = int(seconds_left % 60)
    if hours > 0:
        return f"ETA={hours}h{minutes}m"
    if minutes > 0:
        return f"ETA={minutes}m{seconds}s"
    return f"ETA={seconds}s"


def format_elapsed(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h{minutes}m"
    if minutes > 0:
        return f"{minutes}m{secs}s"
    return f"{secs}s"


def count_files(directory: str) -> int:
    if not os.path.exists(directory):
        return 0
    count = 0
    for _, _, files in os.walk(directory):
        count += len(files)
    return count


def count_warc_files(directory: str) -> int:
    warc_files = glob.glob(os.path.join(directory, "*.warc"))
    warc_gz_files = glob.glob(os.path.join(directory, "*.warc.gz"))
    return len(warc_files) + len(warc_gz_files)


def worker_count(default: int = 4, hard_max: int = 16) -> int:
    """Pick worker count from SLURM_CPUS_PER_TASK or CPU count, capped to avoid oversubscription."""
    try:
        env_val = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    except ValueError:
        env_val = 0
    cpus = env_val if env_val > 0 else (os.cpu_count() or default)
    return max(1, min(cpus, hard_max))


def process_warc_file(warc_file: str, file_type: str, output_dir: str) -> int:
    """Run warc_extractor on a single WARC file; suppress verbose output."""
    warc_dir = os.path.dirname(warc_file)
    warc_name = os.path.basename(warc_file)

    if file_type == "HTML":
        cmd = (
            f"python -u warc_extractor.py http:content-type:text/html "
            f"-dump content -error -path {warc_dir} -string '^{warc_name}$' "
            f"-output_path {output_dir}"
        )
    else:  # PDF
        cmd = (
            f"python -u warc_extractor.py http:content-type:pdf "
            f"-dump content -path {warc_dir} -string '^{warc_name}$' "
            f"-output_path {output_dir}"
        )

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode


def check_paths() -> bool:
    print("Checking paths...")

    if not os.path.exists(WARC_INPUT_DIR):
        print(f"ERROR: WARC input directory not found: {WARC_INPUT_DIR}")
        print("You may need to request access or check the path.")
        return False

    return True


def process_warc_files():
    """Extract HTML then PDF from all WARC files."""
    print("=" * 60)
    print("WARC extraction starting")
    print("=" * 60)

    if not check_paths():
        return

    print("\nFinding WARC files...")
    warc_files = glob.glob(os.path.join(WARC_INPUT_DIR, "*.warc"))
    warc_gz_files = glob.glob(os.path.join(WARC_INPUT_DIR, "*.warc.gz"))
    all_warc_files = sorted(warc_files + warc_gz_files)
    total_warc_files = len(all_warc_files)
    print(f"Found {total_warc_files:,} WARC files to process\n")

    html_output_dir = f"{COLL}_html_files"
    pdf_output_dir = f"{COLL}_pdf_files"
    os.makedirs(html_output_dir, exist_ok=True)
    os.makedirs(pdf_output_dir, exist_ok=True)

    batch_size = 5
    workers = worker_count()
    print(f"Using {workers} parallel workers", flush=True)

    def run_phase(phase_name: str, file_type: str, output_dir: str) -> float:
        print("=" * 80, flush=True)
        print(f"{phase_name}", flush=True)
        print("=" * 80, flush=True)

        start_phase = time.perf_counter()
        batch_start = start_phase
        completed = 0
        total = total_warc_files

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_file = {
                executor.submit(process_warc_file, warc_file, file_type, output_dir): warc_file
                for warc_file in all_warc_files
            }

            for future in as_completed(future_to_file):
                # We ignore per-file errors to keep output minimal
                completed += 1
                if completed % batch_size == 0 or completed == total:
                    now = time.perf_counter()
                    batch_elapsed = now - batch_start
                    total_elapsed = now - start_phase
                    files_this_batch = batch_size if completed % batch_size == 0 else (completed % batch_size)
                    speed = files_this_batch / batch_elapsed if batch_elapsed > 0 else 0
                    remaining = total - completed
                    eta_seconds = remaining / speed if speed > 0 else -1
                    print(
                        f"{completed}/{total} in t={format_elapsed(total_elapsed)} | "
                        f"speed={speed:.2f} files/s | {format_eta(eta_seconds)}",
                        flush=True,
                    )
                    batch_start = time.perf_counter()

        phase_elapsed = time.perf_counter() - start_phase
        return phase_elapsed

    # Phase 1: HTML
    html_time = run_phase("PHASE 1: Extracting HTML files", "HTML", html_output_dir)
    final_html = count_files(html_output_dir)
    print(f"\n✓ HTML done: {final_html:,} files in {format_elapsed(html_time)}\n", flush=True)

    # Phase 2: PDF
    pdf_time = run_phase("PHASE 2: Extracting PDF files", "PDF", pdf_output_dir)
    final_pdf = count_files(pdf_output_dir)
    print(f"\n✓ PDF done: {final_pdf:,} files in {format_elapsed(pdf_time)}\n", flush=True)

    total_time = time.time() - (start_time)
    print(f"\n{'=' * 60}")
    print("✓ Extraction complete!")
    print(f"{'=' * 60}")
    print(f"  - HTML files extracted: {final_html:,}")
    print(f"  - PDF files extracted: {final_pdf:,}")
    print(f"  - Total time: {format_elapsed(total_time)}")
    print(f"\nNext step (Task 4.2): preprocess as needed.")


if __name__ == "__main__":
    print("=" * 60)
    print("WARC Extraction Script Starting")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Script location: {__file__}")
    print()

    if not os.path.exists("warc_extractor.py"):
        print("ERROR: warc_extractor.py not found!")
        print(f"Current directory contents: {os.listdir('.')}")
        sys.exit(1)

    print("✓ warc_extractor.py found")
    print("✓ Starting WARC processing...\n")

    try:
        process_warc_files()
    except Exception as e:
        print(f"\nERROR: Exception occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

