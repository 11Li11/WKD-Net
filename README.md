# WKD-Net

Official model implementation accompanying the manuscript "A frequency-aware
state-space model with transport-residual decoupling for meteorological field
nowcasting."

## Scope of this repository

This repository provides the proposed WKD-Net architecture and a synthetic-input
smoke test. Large meteorological datasets and trained checkpoints are not
distributed because of their size and, for LAPS, data-access restrictions.

## Environment

The experiments were conducted with Python 3.10, PyTorch 2.1.2, CUDA 11.8, and
an NVIDIA RTX 4090 GPU. Create an environment and install the Python packages:

```bash
pip install -r requirements.txt
```

The two-dimensional selective-scan CUDA extension used by the state-space block
is included under `model/lib_mamba/kernels/selective_scan`. On a CUDA system,
install it with:

```bash
pip install ./model/lib_mamba/kernels/selective_scan
```

## Model interface

WKD-Net accepts a tensor with shape `[batch, input_frames, height, width]` and
returns `[batch, forecast_frames, height, width]`.

```python
import torch
from models import WKD_Net

model = WKD_Net(predicted_frames=3, input_frames=5).eval()
x = torch.randn(1, 5, 128, 128)
with torch.no_grad():
    y = model(x)
print(y.shape)
```

Run the interface check with:

```bash
python smoke_test.py --input-frames 5 --output-frames 3 --height 128 --width 128
```

## Dataset configurations

- LAPS: 5 input fields and 3 forecast fields; original grids are cropped to
  256 by 256. The data are subject to institutional access restrictions.
- CIKM-2017: 5 input radar fields and 10 forecast fields. Each 101 by 101 field
  is zero-padded to 128 by 128 before inference. Predictions must be cropped
  back to the original 101 by 101 domain before evaluation.
- ERA5-EastAsia: 6 input precipitation fields and 6 forecast fields at a
  spatial size of 256 by 256. ERA5 data can be downloaded from the Copernicus
  Climate Data Store.

Dataset paths are intentionally not embedded in the model implementation. Users
should prepare tensors in the interface described above using their local data
locations.

## Notes

The repository does not include the comparison-model implementations, raw
datasets, or large checkpoints. Evaluation definitions and experimental
settings are reported in the manuscript.
