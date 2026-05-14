"""DICOM CT/RTSTRUCT input helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import SimpleITK as sitk
from rt_utils import RTStructBuilder


@dataclass(frozen=True)
class ImageMaskData:
    ct_image: sitk.Image
    ct_array: np.ndarray
    tumor_mask: np.ndarray
    oar_masks: list[np.ndarray]
    spacing_mm: tuple[float, float, float]
    origin_mm: tuple[float, float, float]
    direction: tuple[float, ...]


def read_ct_series(ct_dir: str | Path) -> tuple[sitk.Image, np.ndarray]:
    ct_dir = str(ct_dir)
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(ct_dir)
    if not series_ids:
        raise ValueError(f"No DICOM CT series found in {ct_dir}")
    files = reader.GetGDCMSeriesFileNames(ct_dir, series_ids[0])
    reader.SetFileNames(files)
    image = reader.Execute()
    return image, sitk.GetArrayFromImage(image)


def _roi_mask(rtstruct, roi_name: str) -> np.ndarray:
    names = rtstruct.get_roi_names()
    if roi_name not in names:
        raise ValueError(f"ROI '{roi_name}' not found. Available ROIs: {names}")
    # rt-utils returns mask indexed as rows, columns, slices: y,x,z. Convert to z,y,x.
    mask_yxz = rtstruct.get_roi_mask_by_name(roi_name).astype(bool)
    return np.transpose(mask_yxz, (2, 0, 1))


def load_case(ct_dir: str | Path, rtstruct_path: str | Path, tumor_roi: str, oar_rois: Sequence[str] | None = None) -> ImageMaskData:
    image, arr = read_ct_series(ct_dir)
    rtstruct = RTStructBuilder.create_from(dicom_series_path=str(ct_dir), rt_struct_path=str(rtstruct_path))
    tumor = _roi_mask(rtstruct, tumor_roi)
    oars = [_roi_mask(rtstruct, name) for name in (oar_rois or [])]
    return ImageMaskData(
        ct_image=image,
        ct_array=arr,
        tumor_mask=tumor,
        oar_masks=oars,
        spacing_mm=tuple(float(v) for v in image.GetSpacing()),
        origin_mm=tuple(float(v) for v in image.GetOrigin()),
        direction=tuple(float(v) for v in image.GetDirection()),
    )
