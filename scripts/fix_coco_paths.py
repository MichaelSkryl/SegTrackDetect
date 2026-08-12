#!/usr/bin/env python3
"""
Rewrite the `file_name` entries of a COCO annotation file so they resolve where
the dataset is actually mounted.

SegTrackDetect uses `file_name` directly as a filesystem path — `DirectoryDataset`
keeps only the entries for which `os.path.isfile()` is true. Annotations exported
on another machine, or from a dataset that has since moved, therefore load as an
empty dataset rather than raising a useful error.

Run this INSIDE the container, from /SegTrackDetect, so that the absolute paths
it writes are the paths the container will see:

    docker compose run --rm segtrack python scripts/fix_coco_paths.py \\
        --data_root data/MyDataset

Check first with --dry-run, which reports what would change without writing.
The original file is kept as <split>.json.bak.

Images are matched by (sequence directory, file name), so frames with the same
name in different sequences are handled correctly.
"""

import argparse
import json
import os
import shutil
from collections import defaultdict
from glob import glob


IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def index_images(images_dir):
    """Index every image under images_dir by (parent directory, file name).

    Args:
        images_dir (str): Path to the dataset's `images` directory.

    Returns:
        tuple: (by_pair, by_name) where by_pair maps (parent, name) -> path and
            by_name maps name -> list of paths.
    """
    by_pair = {}
    by_name = defaultdict(list)

    for root, _, files in os.walk(images_dir):
        parent = os.path.basename(root)
        for name in files:
            if not name.lower().endswith(IMG_EXTS):
                continue
            path = os.path.join(root, name)
            by_pair[(parent, name)] = path
            by_name[name].append(path)

    return by_pair, by_name


def resolve(file_name, by_pair, by_name):
    """Find the real path of an annotation entry.

    Args:
        file_name (str): The `file_name` value from the annotations.
        by_pair (dict): Index from index_images.
        by_name (dict): Index from index_images.

    Returns:
        tuple: (path, reason) — path is None when the image could not be
            resolved unambiguously, and reason explains why.
    """
    normalised = file_name.replace('\\', '/')
    name = os.path.basename(normalised)
    parent = os.path.basename(os.path.dirname(normalised))

    if (parent, name) in by_pair:
        return by_pair[(parent, name)], 'matched sequence and file name'

    candidates = by_name.get(name, [])
    if len(candidates) == 1:
        return candidates[0], 'matched file name'
    if len(candidates) > 1:
        return None, f'ambiguous — {len(candidates)} files share this name'
    return None, 'no file with this name under images/'


def fix_split(json_path, images_dir, absolute, dry_run):
    """Rewrite one annotation file in place.

    Args:
        json_path (str): Path to the COCO json.
        images_dir (str): Path to the dataset's `images` directory.
        absolute (bool): Write absolute paths rather than paths relative to the
            current working directory.
        dry_run (bool): Report only, change nothing.

    Returns:
        bool: True when every image resolved.
    """
    with open(json_path) as f:
        coco = json.load(f)

    entries = coco.get('images', [])
    if not entries:
        print(f"  {os.path.basename(json_path)}: no 'images' key, skipped")
        return True

    by_pair, by_name = index_images(images_dir)
    if not by_pair:
        print(f"  ERROR: no image files found under {images_dir}")
        return False

    changed, already_ok, failed = 0, 0, []

    for entry in entries:
        old = entry.get('file_name', '')
        path, reason = resolve(old, by_pair, by_name)

        if path is None:
            failed.append((old, reason))
            continue

        new = os.path.abspath(path) if absolute else os.path.relpath(path)
        # Always forward slashes: a json written on a Windows host has to stay
        # usable inside the Linux container.
        new = new.replace('\\', '/')
        if new == old:
            already_ok += 1
        else:
            entry['file_name'] = new
            changed += 1

    label = os.path.basename(json_path)
    print(f"  {label}: {len(entries)} images — "
          f"{changed} rewritten, {already_ok} already correct, "
          f"{len(failed)} unresolved")

    for old, reason in failed[:5]:
        print(f"      unresolved: {old}  ({reason})")
    if len(failed) > 5:
        print(f"      ... and {len(failed) - 5} more")

    if dry_run:
        print("      dry run — nothing written")
    elif changed:
        shutil.copyfile(json_path, json_path + '.bak')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(coco, f, ensure_ascii=False, indent=4)
        print(f"      written (backup: {label}.bak)")

    return not failed


def main():
    parser = argparse.ArgumentParser(
        description='Make COCO file_name entries resolve where the dataset is mounted.')
    parser.add_argument('--data_root', required=True,
                        help='Dataset root, e.g. data/MyDataset')
    parser.add_argument('--split', default=None,
                        help='Single split to fix. Default: every .json in data_root.')
    parser.add_argument('--relative', action='store_true',
                        help='Write paths relative to the current directory instead '
                             'of absolute ones. Survives moving the repository, but '
                             'only works when commands are run from the repo root.')
    parser.add_argument('--dry_run', action='store_true',
                        help='Report what would change without writing.')
    args = parser.parse_args()

    images_dir = os.path.join(args.data_root, 'images')
    if not os.path.isdir(images_dir):
        raise SystemExit(f"No images directory at {images_dir}")

    if args.split:
        splits = [os.path.join(args.data_root, f'{args.split}.json')]
        missing = [p for p in splits if not os.path.isfile(p)]
        if missing:
            raise SystemExit(f"Not found: {missing[0]}")
    else:
        splits = sorted(p for p in glob(os.path.join(args.data_root, '*.json'))
                        if not p.endswith('.bak'))
        if not splits:
            raise SystemExit(f"No .json files in {args.data_root}")

    print(f"Dataset : {os.path.abspath(args.data_root)}")
    print(f"Writing : {'absolute' if not args.relative else 'relative'} paths")
    print(f"Splits  : {len(splits)}")

    ok = all(fix_split(p, images_dir, not args.relative, args.dry_run)
             for p in splits)

    if not ok:
        raise SystemExit("\nSome images could not be resolved — see above. The "
                         "dataset will load only the images that were matched.")
    print("\nAll images resolved.")


if __name__ == '__main__':
    main()
