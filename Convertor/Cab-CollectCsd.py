#!/usr/bin/env python3
"""
Collect all .csd files on the machine and copy them to a given Dataset directory.
By default searches from the user home directory; other search roots can be specified via arguments.
"""

import argparse
import os
import shutil
from pathlib import Path


def find_csd_files(root: Path) -> list[Path]:
    """Recursively find all .csd files under root."""
    found = []
    root = root.resolve()
    try:
        for path in root.rglob("*.csd"):
            if path.is_file():
                found.append(path)
    except PermissionError:
        pass  # Skip directories without permission
    return found


def copy_csd_to_dataset(
    search_root: Path,
    dataset_dir: Path,
    *,
    preserve_structure: bool = False,
    dry_run: bool = False,
) -> tuple[int, int]:
    """
    Find all .csd under search_root and copy to dataset_dir.
    - preserve_structure: If False (default), put all files directly in dataset_dir, adding _1, _2 for duplicates; if True, keep subdirectory structure.
    - Returns (copied_count, skipped_count).
    """
    dataset_dir = dataset_dir.resolve()
    if not dry_run:
        dataset_dir.mkdir(parents=True, exist_ok=True)

    files = find_csd_files(search_root)
    copied = 0
    skipped = 0

    for src in files:
        try:
            rel = src.relative_to(search_root)
        except ValueError:
            rel = src.name

        if preserve_structure:
            dest = dataset_dir / rel
        else:
            dest = dataset_dir / src.name
            if dest.exists() and dest != src.resolve():
                base, ext = os.path.splitext(src.name)
                n = 1
                while dest.exists():
                    dest = dataset_dir / f"{base}_{n}{ext}"
                    n += 1

        if src.resolve() == dest.resolve():
            skipped += 1
            continue

        if dry_run:
            print(f"  [dry-run] {src} -> {dest}")
            copied += 1
            continue

        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            print(f"  Copied: {src} -> {dest}")
            copied += 1
        except OSError as e:
            print(f"  Skip {src}: {e}")
            skipped += 1

    return copied, skipped


def main():
    default_dataset = Path("/Users/lishi/Desktop/Research/CStore/Dataset")
    default_search = Path.home()

    p = argparse.ArgumentParser(description="Find and copy all .csd files to Dataset directory")
    p.add_argument(
        "--search-root",
        type=Path,
        default=default_search,
        help=f"Search root directory, default: {default_search}",
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=default_dataset,
        help=f"Target Dataset directory, default: {default_dataset}",
    )
    p.add_argument(
        "--preserve-structure",
        action="store_true",
        help="Preserve subdirectory structure; default is flat, all .csd in Dataset root only",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print files that would be copied, do not copy",
    )
    args = p.parse_args()

    search_root = args.search_root.resolve()
    if not search_root.is_dir():
        print(f"Error: search root does not exist: {search_root}")
        return 1

    print(f"Search root: {search_root}")
    print(f"Target dir: {args.dataset}")
    print(f"Mode: {'preserve structure' if args.preserve_structure else 'Dataset root only (no subfolders)'}")
    if args.dry_run:
        print("(dry run, no copy)")
    print()

    copied, skipped = copy_csd_to_dataset(
        search_root,
        args.dataset,
        preserve_structure=args.preserve_structure,
        dry_run=args.dry_run,
    )

    print()
    print(f"Done: copied {copied} files, skipped {skipped}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
