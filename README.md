# CornerCrop

[![CI](https://github.com/XucroYuri/CornerCrop/actions/workflows/ci.yml/badge.svg)](https://github.com/XucroYuri/CornerCrop/actions/workflows/ci.yml)

macOS Vision-powered branding text detector, cropper, and verifier.

CornerCrop uses Apple's native Vision.framework OCR to detect removable branding text and crop it out entirely on-device. It is built for both corner copyright lines and cover-page branding blocks such as known brand keywords, issue IDs, and site URLs.

## Features

- **macOS-native OCR** — Uses Vision.framework for fast, local text detection
- **Branding-aware rules** — Matches copyright text, known brand keywords, URLs, and issue IDs before cropping
- **Auto cover detection** — Switches between standard strip crops and more aggressive cover-page crops
- **Verification pass** — Re-runs OCR on processed output and reports residual branding text
- **Batch support** — Accepts multiple files or whole directories
- **Flexible output modes** — Save as `*_nowm`, write to a target directory, or overwrite in place with backups

## Requirements

- macOS 12.0+ (Monterey or later)
- Python 3.9+
- PyObjC (auto-installed with the package)

## Installation

```bash
pip install cornercrop
```

Or from source:

```bash
git clone https://github.com/XucroYuri/CornerCrop.git
cd CornerCrop
python -m pip install -e ".[dev]"
```

## Usage

### Basic

```bash
# Detect, crop, and verify branding removal
cornercrop photo.jpg

# Dry run with verification only
cornercrop photo.jpg --dry-run

# Force aggressive cover-page mode
cornercrop photo.jpg --profile cover
```

### Batch Processing

```bash
# Process explicit files
cornercrop img1.jpg img2.png img3.jpeg

# Process every supported image in a directory
cornercrop ./photos --profile auto

# Save batch outputs into a separate directory
cornercrop ./photos --profile auto --output-dir ./cropped

# Overwrite originals in place
cornercrop ./photos --profile auto --in-place

# Overwrite originals and keep a backup copy
cornercrop ./photos --profile auto --in-place --backup-dir ./originals-backup

# Fail scripts if any residual branding remains after verification
cornercrop ./photos --profile auto --verify --fail-on-residual

# Write a machine-readable report
cornercrop ./photos --report-json ./cornercrop-report.json
```

### Large NAS Album Run

For large NAS-backed image libraries, use the album-level runner instead of one huge recursive CLI
invocation. It scans one top-level segment at a time, processes albums in parallel with resource
backpressure, writes local resume state, and moves unsafe images into an album-local
`_cornercrop_archive/` folder.

```bash
# Smoke-test decisions without modifying images
cornercrop-library /path/to/image-library \
  --state-dir runs/library-watermark/smoke \
  --dry-run --max-album-workers 1

# Start the long in-place run with conservative NAS I/O pressure
scripts/start_library_cornercrop.sh /path/to/image-library

# Use a faster local-Mac/NAS balance when the machine has headroom
RESOURCE_PROFILE=balanced MAX_ALBUM_WORKERS=4 scripts/start_library_cornercrop.sh /path/to/image-library

# Poll progress
scripts/watch_library_cornercrop.py --watch --interval 60

# Verify the DB state against the current non-archive image tree after completion
scripts/audit_library_cornercrop.py /path/to/image-library

# Gracefully stop after the current image
touch runs/library-watermark/live/STOP
```

Policy: images with no detected watermark are skipped; corner watermarks are cropped in place when
the retained image area stays above 70%; non-corner branding or crops that would remove more than
30% of image area are moved into `_cornercrop_archive/`.

Second-pass recovery for archived non-corner branding is available through
`cornercrop-recover-non-corner` and `scripts/start_non_corner_recovery.sh`. It evaluates
multiple crop profiles, verifies the cropped image again, and only moves clean results back to the
parent album. See `docs/large-library-watermark-runbook.md` for the production runbook and audit
checklist.

### Options

```text
positional arguments:
  input                Input image path(s) or directory path(s)

options:
  --output, -o         Output image path (single image mode only)
  --output-dir         Output directory for batch or directory mode
  --in-place           Overwrite cropped images back to the source files
  --backup-dir         Backup directory for originals when using --in-place
  --profile            Crop profile: auto (default), strip, cover, or corner
  --margin N           Extra margin in pixels around detected branding text
  --corner FRACTION    Legacy corner region fraction for corner mode
  --min-confidence F   Minimum OCR confidence threshold
  --max-crop FRACTION  Maximum fraction of any edge to crop
  --verify             Verify processed images for residual branding text (default: on)
  --no-verify          Skip post-processing verification
  --fail-on-residual   Return non-zero if verification still finds branding text
  --report-json PATH   Write summary + per-image results to JSON
  --dry-run            Detect and simulate crop only
  --heartbeat-interval Progress heartbeat interval while long OCR tasks are in flight
  --json               Output per-image results as JSON
  --version            Show version
```

### Python API

```python
from cornercrop.pipeline import process_image
from cornercrop.cropper import CropProfile

result = process_image(
    "photo.jpg",
    strategy=CropProfile.AUTO,
    margin=10,
    verify=True,
    dry_run=True,
)

print(f"Branding candidates: {len(result.branding_candidates)}")
print(f"Selected profile: {result.selected_profile}")
print(f"Verification: {result.verification_status}")
```

## Crop Profiles

| Profile | Description | Content Loss | Best For |
|---------|-------------|-------------|----------|
| **auto** (default) | Detects whether the page is a normal shot or a cover page, then chooses strip or cover automatically | Adaptive | Mixed folders |
| **strip** | Removes the single most efficient edge strip for branding text | Low | Standard copyright lines |
| **cover** | Removes larger top/bottom branding zones on information-heavy pages | Higher | Covers, title cards, site-brand pages |
| **corner** | Legacy mode that only crops corner-classified text | Moderate | Backward compatibility |

## Verification Output

- **`verification_status = clean`** — No branding text matched during post-processing OCR
- **`verification_status = residual_branding_detected`** — Branding text was still found after cropping
- **`verification_status = not_run`** — Verification was skipped with `--no-verify`

## Limitations

- **Text branding only** — Graphic or semi-transparent logo marks still require image analysis or inpainting
- **Rectangular crops only** — Cannot do L-shaped or irregular removals
- **macOS only** — Depends on Apple Vision.framework
- **Aggressive cover crops are intentional** — Cover pages may lose more edge content in exchange for removing all branding text

## Development

```bash
# Run the test suite
uv run --extra dev pytest

# Check the command-line entry points
uv run python -m cornercrop.cli --help
uv run python -m cornercrop.library_runner --help
uv run python -m cornercrop.non_corner_recovery --help
```

Local images, run databases, audit exports, and private notes should stay out of Git. See
`docs/privacy-and-local-data.md` before preparing public commits.

## Contributing and Security

Contributions are welcome. Please read `CONTRIBUTING.md` for development workflow and
`SECURITY.md` for vulnerability reporting.

## Related Projects

- [no-watermark](https://github.com/XucroYuri/no-watermark) — Professional batch watermark removal framework with PaddleOCR, inpainting, and benchmarking (Windows/Linux/macOS)

## License

MIT
