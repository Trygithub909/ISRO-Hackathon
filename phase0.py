"""
Phase 0 — Sentinel-2 Data Download + Preprocessing
=====================================================
Downloads Sentinel-2 imagery from Microsoft Planetary Computer (free, no login),
extracts B8 (NIR/IR) and B4+B3+B2 (RGB) bands,
saves them as matched GeoTIFF pairs ready for Phase 1.

Install:
    pip install pystac-client planetary-computer rasterio numpy opencv-python-headless requests tqdm

Usage:
    python phase0_data_download.py --bbox 72.7 21.0 73.0 21.3 --output_dir raw_data
"""

import os
import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

# ── Optional heavy imports (checked at runtime) ──────────────────────────────
try:
    import rasterio
    from rasterio.enums import Resampling
    RASTERIO_OK = True
except ImportError:
    RASTERIO_OK = False

try:
    import pystac_client
    import planetary_computer
    PC_OK = True
except ImportError:
    PC_OK = False

try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False


# ─────────────────────────────────────────────
# 1. Download helpers
# ─────────────────────────────────────────────

def check_dependencies():
    missing = []
    if not RASTERIO_OK:
        missing.append("rasterio")
    if not PC_OK:
        missing.append("pystac-client planetary-computer")
    if not CV2_OK:
        missing.append("opencv-python-headless")
    if missing:
        print("❌ Missing packages. Install with:")
        print(f"   pip install {' '.join(missing)}")
        sys.exit(1)
    print("✅ All dependencies found.")


def search_sentinel2(
    bbox: list,           # [min_lon, min_lat, max_lon, max_lat]
    start_date: str,      # "YYYY-MM-DD"
    end_date: str,
    max_cloud: int = 10,
    max_items: int = 5,
) -> list:
    """
    Search Planetary Computer STAC for Sentinel-2 L2A scenes.
    Returns a list of STAC items sorted by cloud cover (lowest first).
    """
    print(f"\n🔍 Searching Sentinel-2 scenes ...")
    print(f"   BBOX  : {bbox}")
    print(f"   Dates : {start_date} → {end_date}")
    print(f"   Cloud : < {max_cloud}%")

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    results = catalog.search(
        collections = ["sentinel-2-l2a"],
        bbox        = bbox,
        datetime    = f"{start_date}/{end_date}",
        query       = {"eo:cloud_cover": {"lt": max_cloud}},
    )

    items = list(results.get_items())
    if not items:
        print("⚠️  No scenes found. Try wider date range or higher cloud cover.")
        return []

    # Sort by cloud cover
    items.sort(key=lambda x: x.properties.get("eo:cloud_cover", 99))
    items = items[:max_items]

    print(f"   Found {len(items)} scene(s):")
    for it in items:
        cc  = it.properties.get("eo:cloud_cover", "?")
        dt  = it.properties.get("datetime", "?")[:10]
        print(f"     {it.id}  |  date={dt}  |  cloud={cc:.1f}%")

    return items


def download_file(url: str, dest: Path, desc: str = "") -> Path:
    """Stream-download a file with a progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"   [SKIP] Already downloaded: {dest.name}")
        return dest

    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=desc or dest.name, leave=False
    ) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    return dest


def download_band(item, band: str, dest_dir: Path) -> Path:
    """Download a single Sentinel-2 band GeoTIFF."""
    if band not in item.assets:
        raise KeyError(f"Band {band} not in item assets. Available: {list(item.assets.keys())}")

    href = item.assets[band].href
    dest = dest_dir / f"{item.id}_{band}.tif"
    download_file(href, dest, desc=f"{band}")
    return dest


# ─────────────────────────────────────────────
# 2. Band processing
# ─────────────────────────────────────────────

def read_band_as_uint8(tif_path: Path, target_size: tuple = None) -> np.ndarray:
    """
    Read a single-band GeoTIFF, normalise to uint8.
    Sentinel-2 DN values are uint16 (0–10000 typical reflectance).
    """
    with rasterio.open(tif_path) as src:
        data = src.read(1, resampling=Resampling.bilinear)
        if target_size and (src.width, src.height) != target_size:
            from rasterio.transform import from_bounds
            # Resample via numpy resize (simple)
            data = np.array(
                cv2.resize(data.astype(np.float32),
                           target_size,
                           interpolation=cv2.INTER_LINEAR)
            )

    # Clip to valid reflectance range and normalise
    data = np.clip(data, 0, 10000).astype(np.float32)
    data = (data / 10000.0 * 255.0).astype(np.uint8)
    return data


def make_rgb_tif(
    red_path:   Path,
    green_path: Path,
    blue_path:  Path,
    output_path: Path,
) -> Path:
    """Combine R, G, B bands into a 3-channel GeoTIFF."""
    r = read_band_as_uint8(red_path)
    g = read_band_as_uint8(green_path, target_size=(r.shape[1], r.shape[0]))
    b = read_band_as_uint8(blue_path,  target_size=(r.shape[1], r.shape[0]))

    rgb = np.stack([r, g, b], axis=-1)   # (H, W, 3)
    cv2.imwrite(str(output_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return output_path


def make_ir_tif(nir_path: Path, output_path: Path) -> Path:
    """Save NIR band as grayscale PNG."""
    nir = read_band_as_uint8(nir_path)
    cv2.imwrite(str(output_path), nir)
    return output_path


# ─────────────────────────────────────────────
# 3. Main download + preprocess pipeline
# ─────────────────────────────────────────────

def download_and_prepare(
    bbox:        list,
    start_date:  str,
    end_date:    str,
    output_dir:  str,
    max_cloud:   int  = 10,
    max_items:   int  = 5,
):
    """
    Full pipeline:
      1. Search Sentinel-2 scenes
      2. Download B8 (NIR), B4 (Red), B3 (Green), B2 (Blue)
      3. Convert to uint8 PNG pairs  →  ir/ and rgb/ folders
      4. Print Phase 1 command to run next
    """
    check_dependencies()

    out      = Path(output_dir)
    raw_dir  = out / "raw_bands"
    ir_dir   = out / "ir"
    rgb_dir  = out / "rgb"

    for d in [raw_dir, ir_dir, rgb_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Search
    items = search_sentinel2(bbox, start_date, end_date, max_cloud, max_items)
    if not items:
        return

    downloaded_pairs = 0

    for item in items:
        print(f"\n📥 Downloading scene: {item.id}")

        try:
            # Download bands
            b8_path  = download_band(item, "B08",  raw_dir)   # NIR / IR
            b4_path  = download_band(item, "B04",  raw_dir)   # Red
            b3_path  = download_band(item, "B03",  raw_dir)   # Green
            b2_path  = download_band(item, "B02",  raw_dir)   # Blue

            stem = item.id

            # Process IR
            ir_out  = ir_dir  / f"{stem}_ir.png"
            rgb_out = rgb_dir / f"{stem}_rgb.png"

            print(f"   Processing IR  band → {ir_out.name}")
            make_ir_tif(b8_path, ir_out)

            print(f"   Processing RGB band → {rgb_out.name}")
            make_rgb_tif(b4_path, b3_path, b2_path, rgb_out)

            downloaded_pairs += 1
            print(f"   ✅ Pair saved: {stem}")

        except Exception as e:
            print(f"   ❌ Failed for {item.id}: {e}")
            continue

    # Summary
    print(f"\n{'='*55}")
    print(f"✅ Downloaded {downloaded_pairs} IR+RGB pairs")
    print(f"   IR  images → {ir_dir}")
    print(f"   RGB images → {rgb_dir}")
    print(f"\n{'='*55}")
    print("▶  Next step — run Phase 1:")
    print(f"""
    python phase1_dataset_preparation.py \\
      --ir_dir  {ir_dir} \\
      --rgb_dir {rgb_dir} \\
      --output_dir dataset_patches \\
      --patch_size 256 \\
      --stride 128
""")


# ─────────────────────────────────────────────
# 4. Fallback: Manual download guide
# ─────────────────────────────────────────────

def print_manual_guide(output_dir: str):
    """Print step-by-step manual download instructions."""
    print("""
╔══════════════════════════════════════════════════════════╗
║         MANUAL DOWNLOAD GUIDE (No code needed)          ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  OPTION A — Copernicus Open Access Hub (ESA Official)   ║
║  ─────────────────────────────────────────────────────  ║
║  1. Go to: https://scihub.copernicus.eu                 ║
║  2. Register free account                               ║
║  3. Click the map icon → draw box over your city        ║
║  4. Search filters:                                     ║
║       Mission   : Sentinel-2                            ║
║       Product   : S2MSI2A                               ║
║       Cloud %   : [0 TO 10]                             ║
║  5. Download the .zip file                              ║
║  6. Extract → go to GRANULE/.../IMG_DATA/R10m/          ║
║  7. Copy files:                                         ║
║       *_B08_10m.tif  →  raw_data/ir/                   ║
║       *_B04_10m.tif  →  (red)                          ║
║       *_B03_10m.tif  →  (green)                        ║
║       *_B02_10m.tif  →  (blue)                         ║
║  8. Run combine_bands.py (below) to make RGB            ║
║                                                          ║
║  OPTION B — EarthExplorer (USGS, no login for search)  ║
║  ─────────────────────────────────────────────────────  ║
║  1. Go to: https://earthexplorer.usgs.gov               ║
║  2. Draw area on map → Data Sets → Sentinel-2           ║
║  3. Download individual band TIFFs                      ║
║                                                          ║
║  OPTION C — Planetary Computer (Free, easiest)         ║
║  ─────────────────────────────────────────────────────  ║
║  1. Go to: https://planetarycomputer.microsoft.com      ║
║  2. Explore → Sentinel-2 L2A → Select area             ║
║  3. Download assets directly from browser               ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")
    print(f"After downloading, place files in:")
    print(f"  {output_dir}/ir/   ← NIR band (B8) images")
    print(f"  {output_dir}/rgb/  ← RGB composite images")


# ─────────────────────────────────────────────
# 5. Combine bands utility (for manual downloads)
# ─────────────────────────────────────────────

def combine_manual_bands(
    red_tif:   str,
    green_tif: str,
    blue_tif:  str,
    nir_tif:   str,
    output_dir: str,
    name: str = "scene",
):
    """
    Utility for users who downloaded bands manually.
    Run this after downloading from Copernicus Hub.

    Example:
        python phase0_data_download.py --combine \
          --red   downloads/T43RGP_B04_10m.tif \
          --green downloads/T43RGP_B03_10m.tif \
          --blue  downloads/T43RGP_B02_10m.tif \
          --nir   downloads/T43RGP_B08_10m.tif \
          --output_dir raw_data --name scene1
    """
    check_dependencies()

    out     = Path(output_dir)
    ir_dir  = out / "ir"
    rgb_dir = out / "rgb"
    ir_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing bands for: {name}")

    ir_out  = ir_dir  / f"{name}_ir.png"
    rgb_out = rgb_dir / f"{name}_rgb.png"

    make_ir_tif(Path(nir_tif), ir_out)
    print(f"  IR  → {ir_out}")

    make_rgb_tif(Path(red_tif), Path(green_tif), Path(blue_tif), rgb_out)
    print(f"  RGB → {rgb_out}")

    print(f"\n✅ Done. Now run Phase 1:")
    print(f"""
    python phase1_dataset_preparation.py \\
      --ir_dir  {ir_dir} \\
      --rgb_dir {rgb_dir} \\
      --output_dir dataset_patches \\
      --patch_size 256 --stride 128
""")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 0: Download Sentinel-2 data for IR Colorisation"
    )

    # Auto-download mode
    parser.add_argument("--bbox",       nargs=4, type=float,
                        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
                        help="Bounding box. Example: 72.7 21.0 73.0 21.3 (Surat)")
    parser.add_argument("--start_date", default="2023-01-01",
                        help="Start date YYYY-MM-DD (default: 2023-01-01)")
    parser.add_argument("--end_date",   default="2023-12-31",
                        help="End date YYYY-MM-DD (default: 2023-12-31)")
    parser.add_argument("--max_cloud",  type=int,   default=10,
                        help="Maximum cloud cover %% (default: 10)")
    parser.add_argument("--max_items",  type=int,   default=5,
                        help="Max number of scenes to download (default: 5)")
    parser.add_argument("--output_dir", default="raw_data",
                        help="Output directory (default: raw_data)")

    # Manual combine mode
    parser.add_argument("--combine",    action="store_true",
                        help="Combine manually downloaded band TIFFs")
    parser.add_argument("--red",        help="Path to Red band TIF (B04)")
    parser.add_argument("--green",      help="Path to Green band TIF (B03)")
    parser.add_argument("--blue",       help="Path to Blue band TIF (B02)")
    parser.add_argument("--nir",        help="Path to NIR band TIF (B08)")
    parser.add_argument("--name",       default="scene",
                        help="Scene name for output files")

    # Guide mode
    parser.add_argument("--guide",      action="store_true",
                        help="Print manual download guide")

    args = parser.parse_args()

    if args.guide:
        print_manual_guide(args.output_dir)

    elif args.combine:
        if not all([args.red, args.green, args.blue, args.nir]):
            print("❌ --combine requires --red --green --blue --nir")
            sys.exit(1)
        combine_manual_bands(
            args.red, args.green, args.blue, args.nir,
            args.output_dir, args.name,
        )

    elif args.bbox:
        download_and_prepare(
            bbox        = args.bbox,
            start_date  = args.start_date,
            end_date    = args.end_date,
            output_dir  = args.output_dir,
            max_cloud   = args.max_cloud,
            max_items   = args.max_items,
        )

    else:
        print("Usage examples:")
        print()
        print("  # Auto-download (Surat, India):")
        print("  python phase0_data_download.py \\")
        print("    --bbox 72.7 21.0 73.0 21.3 \\")
        print("    --start_date 2023-01-01 --end_date 2023-12-31 \\")
        print("    --output_dir raw_data")
        print()
        print("  # Combine manually downloaded bands:")
        print("  python phase0_data_download.py --combine \\")
        print("    --red B04.tif --green B03.tif --blue B02.tif --nir B08.tif \\")
        print("    --output_dir raw_data --name scene1")
        print()
        print("  # Print manual guide:")
        print("  python phase0_data_download.py --guide")
