"""Export centers, masks, reports, and optional RTSTRUCT contours."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import json
import numpy as np
import pandas as pd
import SimpleITK as sitk
from rt_utils import RTStructBuilder


def write_centers_csv(path: str | Path, centers_mm: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(np.asarray(centers_mm, dtype=float), columns=["x_mm", "y_mm", "z_mm"])
    df.insert(0, "sphere_id", np.arange(1, len(df) + 1))
    df.to_csv(path, index=False)


def write_nifti_mask(path: str | Path, mask_zyx: np.ndarray, reference_image: sitk.Image | None = None, spacing_mm: Sequence[float] | None = None, origin_mm: Sequence[float] | None = None, direction: Sequence[float] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = sitk.GetImageFromArray(np.asarray(mask_zyx, dtype=np.uint8))
    if reference_image is not None:
        img.CopyInformation(reference_image)
    else:
        if spacing_mm is not None:
            img.SetSpacing(tuple(float(v) for v in spacing_mm))
        if origin_mm is not None:
            img.SetOrigin(tuple(float(v) for v in origin_mm))
        if direction is not None:
            img.SetDirection(tuple(float(v) for v in direction))
    sitk.WriteImage(img, str(path))


def write_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def export_rtstruct(ct_dir: str | Path, base_rtstruct: str | Path | None, output_path: str | Path, union_mask_zyx: np.ndarray, individual_masks_zyx: Sequence[np.ndarray]) -> None:
    """Export RTSTRUCT using rt-utils.

    Masks are converted from z,y,x to rt-utils y,x,z. This function is best-effort
    and may fail for sparse or tiny contours depending on the DICOM series geometry.
    """
    if base_rtstruct:
        rtstruct = RTStructBuilder.create_from(dicom_series_path=str(ct_dir), rt_struct_path=str(base_rtstruct))
    else:
        rtstruct = RTStructBuilder.create_new(dicom_series_path=str(ct_dir))
    def yxz(mask):
        return np.transpose(np.asarray(mask, dtype=bool), (1, 2, 0))
    rtstruct.add_roi(mask=yxz(union_mask_zyx), name="SFRT_UNION", color=[255, 0, 0])
    for i, mask in enumerate(individual_masks_zyx, start=1):
        rtstruct.add_roi(mask=yxz(mask), name=f"SFRT_{i:03d}", color=[255, 128, 0])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    rtstruct.save(str(output_path))
