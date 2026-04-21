# CornerCrop

macOS Vision-powered corner watermark detector & cropper.

CornerCrop uses Apple's native Vision.framework OCR to detect text watermarks in image corners and crops them out with minimal content loss. It runs entirely on-device — no cloud APIs, no external OCR engines.

## Features

- **macOS-native OCR** — Leverages Vision.framework for fast, accurate, on-device text detection
- **Corner-aware** — Only targets text in image corners (configurable region), preserving center content
- **Two crop strategies** — Strip (remove shorter edge, less content loss) or Corner (remove both edges, more thorough)
- **Safety guardrails** — Configurable max-crop fraction prevents over-removal
- **Batch support** — Process multiple images in one command
- **Zero cloud dependency** — Everything runs locally on your Mac

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
pip install -e .
```

## Usage

### Basic

```bash
# Detect and crop watermarks
cornercrop photo.jpg

# Dry run (detect only, no output file)
cornercrop photo.jpg --dry-run

# Use corner strategy (crop both edges at watermark corner)
cornercrop photo.jpg --strategy corner
```

### Batch processing

```bash
cornercrop img1.jpg img2.png img3.jpeg
```

### Options

```
positional arguments:
  input                Input image path(s)

options:
  --output, -o         Output image path (single image mode)
  --strategy           Crop strategy: strip (default) or corner
  --margin N           Extra margin pixels around watermark (default: 10)
  --corner FRACTION    Corner region fraction 0-0.5 (default: 0.20)
  --min-confidence F   Minimum OCR confidence threshold (default: 0.3)
  --max-crop FRACTION  Maximum fraction of any edge to crop (default: 0.25)
  --dry-run            Detect only, don't save cropped images
  --json               Output results as JSON
  --version            Show version
```

### Python API

```python
from cornercrop.pipeline import process_image
from cornercrop.cropper import CropStrategy

result = process_image(
    "photo.jpg",
    strategy=CropStrategy.STRIP,
    corner_frac=0.20,
    margin=10,
    dry_run=True,  # set False to save cropped image
)

print(f"Watermarks found: {len(result.watermarks)}")
print(f"Output size: {result.output_size}")
```

## How It Works

```
┌─────────────────────────┐
│    Input Image           │
│  ┌───┐           ┌───┐  │
│  │TL │           │TR │  │   TL/TR/BL/BR = corner text
│  └───┘           └───┘  │   detected by Vision OCR
│                          │
│       Main Content       │
│                          │
│  ┌───┐           ┌───┐  │
│  │BL │           │BR │  │
│  └───┘           └───┘  │
└─────────────────────────┘
          │
          ▼
┌─────────────────────────┐
│  Vision OCR Detection    │
│  → Text regions + bbox   │
│  → Corner classification │
└─────────────────────────┘
          │
          ▼
┌─────────────────────────┐
│  Crop Computation        │
│  Strip: remove shorter   │
│         edge strip       │
│  Corner: remove both     │
│          edge strips     │
└─────────────────────────┘
          │
          ▼
    Cropped Output Image
```

## Crop Strategies

| Strategy | Description | Content Loss | Best For |
|----------|-------------|-------------|----------|
| **strip** (default) | Removes the shorter edge strip containing the watermark | Minimal | Watermarks near one edge |
| **corner** | Removes both edge strips at the watermark corner | Moderate | Watermarks exactly at corner intersection |

## Limitations

- **Text watermarks only** — Vision OCR detects text; graphic/semi-transparent logo watermarks require image analysis (planned)
- **Rectangular crops** — Cannot do L-shaped or irregular removals (would require inpainting)
- **macOS only** — Depends on Apple Vision.framework

## Related Projects

- [no-watermark](https://github.com/XucroYuri/no-watermark) — Professional batch watermark removal framework with PaddleOCR, inpainting, and benchmarking (Windows/Linux/macOS)

## License

MIT
