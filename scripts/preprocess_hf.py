"""
preprocess_hf.py

Download LLaVA v1.5 finetuning data via HuggingFace datasets/CDN.

Produces the same layout as preprocess.py but routes through HF where
possible — much faster on networks where HF has good bandwidth but other
hosts (cocodataset.org, cs.stanford.edu) are throttled.

Layout:
    <root_dir>/download/llava-v1.5-instruct/
        llava_v1_5_mix665k.json
        coco/train2017/*.jpg         (from detection-datasets/coco)
        gqa/images/*.jpg             (from lmms-lab/GQA)
        ocr_vqa/images/*.jpg         (from qnguyen3/ocr_vqa — zip on HF)
        textvqa/train_images/*.jpg   (from lmms-lab/textvqa)
        vg/VG_100K/*.jpg             (no HF mirror — manual download)
        vg/VG_100K_2/*.jpg

Usage:
    python scripts/preprocess_hf.py --root_dir /fast/txia/salaadpp
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import urllib.request
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

from huggingface_hub import hf_hub_download
from tqdm import tqdm

from prismatic.overwatch import initialize_overwatch

overwatch = initialize_overwatch(__name__)


def _ensure(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    return d


def _n_files(d: Path) -> int:
    return sum(1 for _ in d.iterdir()) if d.exists() else 0


HF_CACHE = "/fast/txia/salaadpp/.hf_cache"


def _extract_images(
    repo_id: str,
    split: str,
    dest_dir: Path,
    filename_fn,
    desc: str,
    total: int | None = None,
    config: str | None = None,
    only_fnames: set[str] | None = None,
) -> int:
    import datasets as ds_lib
    from datasets import load_dataset

    _ensure(dest_dir)
    overwatch.info(f"    downloading parquet from HF (bulk) ...")
    ds = load_dataset(repo_id, name=config, split=split, cache_dir=HF_CACHE, num_proc=4)
    ds = ds.cast_column("image", ds_lib.Image(decode=False))
    overwatch.info(f"    writing {len(ds)} rows → {dest_dir}")
    saved = 0
    seen: set[str] = set()
    for ex in tqdm(ds, desc=desc, total=total or len(ds)):
        fname = filename_fn(ex)
        if fname is None or fname in seen:
            continue
        seen.add(fname)
        if only_fnames is not None and fname not in only_fnames:
            continue
        p = dest_dir / fname
        if not p.exists():
            raw = ex["image"]["bytes"]
            with open(p, "wb") as f:
                f.write(raw)
            saved += 1
    return saved


_LEAF_DIRS = {
    "coco": ["coco/train2017"],
    "gqa": ["gqa/images"],
    "ocr_vqa": ["ocr_vqa/images"],
    "textvqa": ["textvqa/train_images"],
    "vg": ["vg/VG_100K", "vg/VG_100K_2"],
}


def _find_missing(dest: Path) -> dict[str, set[str]]:
    """Parse annotations and return missing image paths grouped by dataset prefix."""
    ann = dest / "llava_v1_5_mix665k.json"
    data = json.loads(ann.read_text())

    referenced: dict[str, set[str]] = defaultdict(set)
    for item in data:
        img = item.get("image", "")
        if img:
            referenced[img.split("/")[0]].add(img)

    existing: set[str] = set()
    for prefix in referenced:
        for leaf in _LEAF_DIRS.get(prefix, []):
            leaf_path = dest / leaf
            if leaf_path.is_dir():
                for fname in os.listdir(leaf_path):
                    existing.add(f"{leaf}/{fname}")

    by_prefix: dict[str, set[str]] = {}
    for prefix, paths in referenced.items():
        missing = paths - existing
        if missing:
            by_prefix[prefix] = missing
    return by_prefix


# ---------- individual datasets ----------


def download_annotations(dest: Path) -> None:
    out = dest / "llava_v1_5_mix665k.json"
    if out.exists():
        overwatch.info(f"  annotations: {out} (exists)")
        return
    overwatch.info("  annotations: HF → liuhaotian/LLaVA-Instruct-150K")
    hf_hub_download(
        repo_id="liuhaotian/LLaVA-Instruct-150K",
        filename="llava_v1_5_mix665k.json",
        repo_type="dataset",
        local_dir=dest,
    )


_COCO_INDIVIDUAL_THRESHOLD = 5000


def download_coco(dest: Path, missing: set[str]) -> None:
    """COCO train2017 — individual fetch for small gaps, bulk zip otherwise."""
    img_dir = dest / "coco" / "train2017"

    if not missing:
        overwatch.info("  COCO: nothing missing, skipping")
        return

    if len(missing) <= _COCO_INDIVIDUAL_THRESHOLD:
        _ensure(img_dir)
        overwatch.info(f"  COCO: fetching {len(missing)} missing images individually")
        failed = []
        for rel in tqdm(sorted(missing), desc="COCO"):
            fname = Path(rel).name
            url = f"http://images.cocodataset.org/train2017/{fname}"
            out = dest / rel
            if out.exists():
                continue
            try:
                urllib.request.urlretrieve(url, str(out))
            except Exception as e:
                failed.append((fname, e))
        if failed:
            overwatch.warning(f"  COCO: {len(failed)} images failed to download")
            for fname, e in failed[:5]:
                overwatch.warning(f"    {fname}: {e}")
        overwatch.info(f"  COCO: {_n_files(img_dir)} images")
        return

    from pySmartDL import SmartDL

    overwatch.info(f"  COCO: {len(missing)} missing (>{_COCO_INDIVIDUAL_THRESHOLD}), using bulk zip")
    zip_path = dest / "train2017.zip"
    url = "http://images.cocodataset.org/zips/train2017.zip"

    if not zip_path.exists():
        overwatch.info(f"  COCO: downloading {url} (16 connections)")
        dl = SmartDL(url, str(zip_path), threads=16)
        dl.start()

    missing_fnames = {Path(p).name for p in missing}
    with ZipFile(zip_path) as zf:
        to_extract = [m for m in zf.namelist() if not m.endswith("/") and Path(m).name in missing_fnames]
        overwatch.info(f"  COCO: extracting {len(to_extract)} missing files from zip")
        _ensure(img_dir)
        zf.extractall(dest / "coco", members=to_extract)

    zip_path.unlink()
    overwatch.info(f"  COCO: {_n_files(img_dir)} images")


def download_gqa(dest: Path, missing: set[str]) -> None:
    """GQA — ~100K images via lmms-lab/GQA on HF (multiple image splits)."""
    if not missing:
        overwatch.info("  GQA: nothing missing, skipping")
        return

    img_dir = dest / "gqa" / "images"
    overwatch.info(f"  GQA: HF → lmms-lab/GQA ({len(missing)} missing)")
    configs = [
        "train_all_images",
        "val_all_images",
        "test_all_images",
        "testdev_all_images",
        "submission_all_images",
        "challenge_all_images",
    ]
    needed_fnames = {Path(p).name for p in missing}
    total = 0
    for cfg in configs:
        split = cfg.replace("_all_images", "")
        n = _extract_images(
            "lmms-lab/GQA",
            split,
            img_dir,
            filename_fn=lambda ex: (
                ex["id"] if ex["id"].endswith(".jpg") else f"{ex['id']}.jpg"
            ),
            desc=f"GQA {cfg}",
            config=cfg,
            only_fnames=needed_fnames,
        )
        total += n
        needed_fnames -= {f.name for f in img_dir.iterdir()}
        if not needed_fnames:
            break
    overwatch.info(f"  GQA: saved {total} images total")


def download_ocrvqa(dest: Path, missing: set[str]) -> None:
    """OCR-VQA — zip from qnguyen3/ocr_vqa on HF CDN."""
    if not missing:
        overwatch.info("  OCR-VQA: nothing missing, skipping")
        return

    img_dir = dest / "ocr_vqa" / "images"
    overwatch.info(f"  OCR-VQA: HF → qnguyen3/ocr_vqa (zip, {len(missing)} missing)")
    cached_zip = hf_hub_download(
        repo_id="qnguyen3/ocr_vqa",
        filename="ocr_vqa.zip",
        repo_type="dataset",
    )

    needed_fnames = {Path(p).name for p in missing}
    needed_stems = {Path(p).stem for p in needed_fnames}

    tmp = dest / "_ocr_vqa_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)

    _ensure(img_dir)
    with ZipFile(cached_zip) as zf:
        members = [m for m in zf.namelist() if not m.endswith("/") and Path(m).stem in needed_stems]
        overwatch.info(f"  OCR-VQA: extracting {len(members)} matching members from zip")
        zf.extractall(tmp, members=members)

    extracted_dirs = [d for d in tmp.iterdir() if d.is_dir()]
    src = extracted_dirs[0] if len(extracted_dirs) == 1 else tmp
    for f in src.iterdir():
        if f.is_file():
            shutil.move(str(f), str(img_dir / f.name))
    if tmp.exists():
        shutil.rmtree(tmp)

    overwatch.info("  OCR-VQA: converting GIF/PNG → JPG")
    from prismatic.preprocessing import convert_to_jpg

    convert_to_jpg(img_dir)


def download_textvqa(dest: Path, missing: set[str]) -> None:
    """TextVQA — ~29K unique images via lmms-lab/textvqa on HF."""
    if not missing:
        overwatch.info("  TextVQA: nothing missing, skipping")
        return

    img_dir = dest / "textvqa" / "train_images"
    overwatch.info(f"  TextVQA: HF → lmms-lab/textvqa ({len(missing)} missing)")
    needed_fnames = {Path(p).name for p in missing}
    total = 0
    for split in ["train", "validation"]:
        n = _extract_images(
            "lmms-lab/textvqa",
            split,
            img_dir,
            filename_fn=lambda ex: f"{ex['image_id']}.jpg",
            desc=f"TextVQA {split}",
            only_fnames=needed_fnames,
        )
        total += n
        needed_fnames -= {f.name for f in img_dir.iterdir()}
        if not needed_fnames:
            break
    overwatch.info(f"  TextVQA: saved {total} images total")


def download_vg(dest: Path, missing: set[str]) -> None:
    """Visual Genome — no HF mirror. Multi-connection download via pySmartDL."""
    if not missing:
        overwatch.info("  VG: nothing missing, skipping")
        return

    from pySmartDL import SmartDL

    needed_fnames = {Path(p).name for p in missing}

    for url, name in [
        ("https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip", "vg/VG_100K"),
        ("https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip", "vg/VG_100K_2"),
    ]:
        if not needed_fnames:
            break
        target = dest / name

        zip_path = dest / Path(url).name
        if not zip_path.exists():
            overwatch.info(f"  VG: downloading {url} (16 connections)")
            dl = SmartDL(url, str(zip_path), threads=16)
            dl.start()

        _ensure(target)
        with ZipFile(zip_path) as zf:
            members = [m for m in zf.namelist() if not m.endswith("/") and Path(m).name in needed_fnames]
            overwatch.info(f"  VG: extracting {len(members)} missing members from {zip_path.name}")
            for m in tqdm(members, desc=f"VG {name}"):
                data = zf.read(m)
                with open(target / Path(m).name, "wb") as f:
                    f.write(data)
            needed_fnames -= {Path(m).name for m in members}
        zip_path.unlink()
    overwatch.info(f"  VG: done")


# ---------- main ----------


def main() -> None:
    parser = argparse.ArgumentParser(description="Download LLaVA v1.5 data via HuggingFace")
    parser.add_argument("--root_dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    dest = args.root_dir / "download" / "llava-v1.5-instruct"
    _ensure(dest)
    overwatch.info(f"LLaVA v1.5 data → {dest}")

    download_annotations(dest)

    by_prefix = _find_missing(dest)
    total_missing = sum(len(v) for v in by_prefix.values())
    if total_missing == 0:
        overwatch.info("All images present — nothing to download.")
        return

    overwatch.info(f"Missing {total_missing} images across {len(by_prefix)} datasets:")
    for prefix, paths in sorted(by_prefix.items()):
        overwatch.info(f"  {prefix}: {len(paths)}")

    download_coco(dest, by_prefix.get("coco", set()))
    download_gqa(dest, by_prefix.get("gqa", set()))
    download_ocrvqa(dest, by_prefix.get("ocr_vqa", set()))
    download_textvqa(dest, by_prefix.get("textvqa", set()))
    download_vg(dest, by_prefix.get("vg", set()))

    remaining = _find_missing(dest)
    still_missing = sum(len(v) for v in remaining.values())
    if still_missing:
        overwatch.warning(f"Still missing {still_missing} images after augment")
    else:
        overwatch.info("Done — all referenced images now present.")


if __name__ == "__main__":
    main()
