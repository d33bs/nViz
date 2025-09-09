"""
Experiment with GFF image stacks to OME-ZARR with display in Napari.
"""

import os
import pathlib
from itertools import groupby
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tifffile as tiff
import vtk
import zarr
from ome_zarr.io import parse_url as zarr_parse_url
from ome_zarr.writer import write_image as zarr_write_image
from vtkmodules.util import numpy_support as ns

from .image_meta import extract_z_slice_number_from_filename, generate_ome_xml


def image_set_to_arrays(
    image_dir: str,
    channel_map: Dict[str, str],
    label_dir: Optional[str] = None,
    ignore: Optional[List[str]] = ["Merge"],
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Read a set of images as an array of images.
    We follow a convention of splitting the following
    into separate nested dictionaries for use by
    other functions within this project.

    - "images": original images
    - "labels": images which represent objects of interest
        within the original "images"

    Args:
        image_dir (str):
            Directory containing TIFF image files.
        channel_map (Dict[str, str]):
            Mapping from filename codes to channel names.
        label_dir (Optional[str]):
            Directory containing label TIFF files. Defaults to None.
        ignore (Optional[List[str]]):
            List of filename codes to ignore.
            Defaults to ["Merge"], which is a
            code for merged images.

    Returns:
        Dict[str, Dict[str, np.ndarray]]:
            A dictionary containing two keys: "images" and "labels".
            Each key maps to another dictionary where the keys are
            channel names and the values are numpy arrays of images.
    """
    # build a reference to the observations
    zstack_arrays = {
        "images": {
            channel_map.get(filename_code, filename_code): np.stack(
                [
                    tiff.imread(tiff_file.path).astype(np.uint16)
                    for tiff_file in sorted(
                        files,
                        key=lambda x: extract_z_slice_number_from_filename(x.name),
                    )
                ]
            ).astype(np.uint16)
            for filename_code, files in groupby(
                sorted(
                    [
                        file
                        for file in os.scandir(image_dir)
                        if (file.name.endswith(".tif") or file.name.endswith(".tiff"))
                        and (
                            file.name.split("_")[1] not in ignore
                            if ignore is not None
                            else True
                        )
                    ],
                    key=lambda x: x.name.split("_")[1],
                ),
                key=lambda x: x.name.split("_")[1],
            )
        }
    }

    if label_dir:
        zstack_arrays["labels"] = {
            f"{pathlib.Path(label_name).stem} (labels)": tiff.imread(
                next(iter(file)).path
            ).astype(np.uint16)
            for label_name, file in groupby(
                sorted(
                    [
                        file
                        for file in os.scandir(label_dir)
                        if (file.name.endswith(".tif") or file.name.endswith(".tiff"))
                        and (
                            file.name.split("_")[0] not in ignore
                            if ignore is not None
                            else True
                        )
                    ],
                    key=lambda x: x.name.split("_")[0],
                ),
                key=lambda x: x.name.split("_")[0],
            )
        }

    return zstack_arrays


def tiff_to_zarr(  # noqa: PLR0913
    image_dir: str,
    output_path: str,
    channel_map: Dict[str, str],
    scaling_values: Union[List[int], Tuple[int]],
    label_dir: Optional[str] = None,
    ignore: Optional[List[str]] = ["Merge"],
) -> str:
    """
    Convert TIFF files to OME-Zarr format.

    Args:
        image_dir (str):
            Directory containing TIFF image files.
        output_path (str):
            Path to save the output OME-Zarr file.
        channel_map (Dict[str, str]):
            Mapping from filename codes to channel names.
        scaling_values (Union[List[int], Tuple[int]]):
            Scaling values for the images.
        label_dir (Optional[str]):
            Directory containing label TIFF files. Defaults to None.
        ignore (Optional[List[str]]):
            List of filename codes to ignore.
            Defaults to ["Merge"], which is a
            code for merged images.

    Returns:
        str: Path to the output OME-Zarr file.
    """

    # except on dir already existing
    if pathlib.Path(output_path).is_dir():
        raise FileExistsError(
            (
                f"Output path {output_path} already exists."
                "Please remove before creating a new Zarr."
            )
        )

    if not pathlib.Path(image_dir).is_dir():
        raise NotADirectoryError(f"Image directory {image_dir} does not exist.")

    # build a reference to the observations
    frame_zstacks = image_set_to_arrays(
        image_dir=image_dir, label_dir=label_dir, channel_map=channel_map, ignore=ignore
    )

    # Parse URL and ensure store is compatible
    store = zarr_parse_url(output_path, mode="w").store
    # Ensure we are working with a Zarr group
    root = zarr.group(store, overwrite=True)

    # create scaling metadata
    scale_metadata = [
        {
            "datasets": [
                {
                    "path": "0",  # Path to the dataset
                    "coordinateTransformations": [
                        {
                            "type": "scale",
                            "scale": list(scaling_values),
                        }  # Apply scaling values
                    ],
                }
            ],
            "axes": [
                {
                    "name": "z",
                    "unit": "micrometer",
                    "type": "space",
                },  # Define the z-axis
                {
                    "name": "y",
                    "unit": "micrometer",
                    "type": "space",
                },  # Define the y-axis
                {
                    "name": "x",
                    "unit": "micrometer",
                    "type": "space",
                },  # Define the x-axis
            ],
        }
    ]

    # Write each channel separately to the Zarr file with no compression
    # Save images to OME-Zarr format
    images_group = root.create_group("images")
    for channel, stack in frame_zstacks["images"].items():
        zarr_write_image(
            image=stack,
            group=(group := images_group.create_group(channel)),
            axes="zyx",  # Specify the axes order for each channel
            dtype="uint16",  # Ensure the dtype is set correctly
            scaler=None,  # Disable scaler
        )
        # Set the units attribute for the group to "micrometers"
        group.attrs["units"] = "micrometers"

        # Define the multiscales metadata for the group
        group.attrs["multiscales"] = scale_metadata

    if label_dir:
        # Save masks to OME-Zarr format
        labels_group = root.create_group("labels")
        for compartment_name, stack in frame_zstacks["labels"].items():
            zarr_write_image(
                image=stack,
                group=(group := labels_group.create_group(compartment_name)),
                axes="zyx",  # Specify the axes order for each mask
                dtype="uint16",  # Ensure the dtype is set correctly
                scaler=None,  # Disable scaler
            )
            # Set the units attribute for the group to "micrometers"
            group.attrs["units"] = "micrometers"

            # Define the multiscales metadata for the group
            group.attrs["multiscales"] = scale_metadata

    return output_path


def tiff_to_ometiff(  # noqa: PLR0913
    image_dir: str,
    output_path: str,
    channel_map: Dict[str, str],
    scaling_values: Union[List[int], Tuple[int]],
    label_dir: Optional[str] = None,
    ignore: Optional[List[str]] = ["Merge"],
) -> str:
    """
    Convert TIFF files to OME-TIFF format.

    Args:
        image_dir (str):
            Directory containing TIFF image files.
        output_path (str):
            Path to save the output OME-TIFF file.
        channel_map (Dict[str, str]):
            Mapping from filename codes to channel names.
        scaling_values (Union[List[int], Tuple[int]]):
            Scaling values for the images.
        label_dir (Optional[str]):
            Directory containing label TIFF files. Defaults to None.
        ignore (Optional[List[str]]):
            List of filename codes to ignore.
            Defaults to ["Merge"], which is a
            code for merged images.

    Returns:
        str: Path to the output OME-TIFF file.
    """

    # except on dir already existing
    if pathlib.Path(output_path).is_file():
        raise FileExistsError(
            (
                f"Output path {output_path} already exists."
                "Please remove before creating a new OME-TIFF."
            )
        )

    if not pathlib.Path(image_dir).is_dir():
        raise NotADirectoryError(f"Image directory {image_dir} does not exist.")

    frame_zstacks = image_set_to_arrays(
        image_dir=image_dir, label_dir=label_dir, channel_map=channel_map, ignore=ignore
    )

    # Prepare the data for writing
    images_data = []
    labels_data = []
    channel_names = []
    label_names = []

    # Collect image data
    for channel, stack in frame_zstacks["images"].items():
        images_data.append(stack)
        channel_names.append(channel)

    # Collect label data
    if label_dir:
        for compartment_name, stack in frame_zstacks["labels"].items():
            labels_data.append(stack)
            label_names.append(compartment_name)

    # Stack the images and labels along a new axis for channels
    images_data = np.stack(images_data, axis=0)
    if labels_data:
        labels_data = np.stack(labels_data, axis=0)
        combined_data = np.concatenate((images_data, labels_data), axis=0)
        combined_channel_names = channel_names + label_names
    else:
        combined_data = images_data
        combined_channel_names = channel_names

    # Generate OME-XML metadata
    ome_metadata = {
        "SizeC": combined_data.shape[0],
        "SizeZ": combined_data.shape[1],
        "SizeY": combined_data.shape[2],
        "SizeX": combined_data.shape[3],
        "PhysicalSizeX": scaling_values[2],
        "PhysicalSizeY": scaling_values[1],
        "PhysicalSizeZ": scaling_values[0],
        # note: we use 7-bit ascii compatible characters below
        # due to tifffile limitations
        "PhysicalSizeXUnit": "um",
        "PhysicalSizeYUnit": "um",
        "PhysicalSizeZUnit": "um",
        "Channel": [{"Name": name} for name in combined_channel_names],
    }
    ome_xml = generate_ome_xml(ome_metadata)

    # Write the combined data to a single OME-TIFF
    with tiff.TiffWriter(output_path, bigtiff=True) as tif:
        tif.write(combined_data, description=ome_xml, photometric="minisblack")

    return output_path


def _np_to_vtk_image(
    arr: np.ndarray,
    spacing: Tuple[float, float, float],
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> vtk.vtkImageData:
    """Convert a (Z, Y, X) NumPy array to vtkImageData with correct geometry.

    Args:
        arr: Volume as (Z, Y, X).
        spacing: Physical voxel spacing as (sz, sy, sx) in micrometers.
        origin: Origin in physical units (default (0,0,0)).

    Returns:
        vtkImageData with scalars set as the active point data.
    """
    if arr.ndim != 3:  # noqa: PLR2004
        raise ValueError(f"Expected a 3D array (Z,Y,X). Got shape {arr.shape}")

    z, y, x = arr.shape
    vtk_img = vtk.vtkImageData()
    vtk_img.SetDimensions(int(x), int(y), int(z))
    vtk_img.SetSpacing(
        float(spacing[2]), float(spacing[1]), float(spacing[0])
    )  # (sx, sy, sz)
    vtk_img.SetOrigin(
        float(origin[2]), float(origin[1]), float(origin[0])
    )  # (ox, oy, oz)

    vtk_arr = ns.numpy_to_vtk(
        num_array=arr.ravel(order="F"),
        deep=True,
        array_type=ns.get_vtk_array_type(arr.dtype),
    )
    vtk_arr.SetName("ImageScalars")
    vtk_img.GetPointData().SetScalars(vtk_arr)
    return vtk_img


def _add_string_field(md_target: vtk.vtkDataObject, name: str, value: str) -> None:
    """Attach a small string metadata field to vtkDataObject.FieldData."""
    sarr = vtk.vtkStringArray()
    sarr.SetName(name)
    sarr.InsertNextValue(value)
    md_target.GetFieldData().AddArray(sarr)


def _discrete_surfaces_from_labels(
    lbl_img: vtk.vtkImageData,
    label_ids: Optional[List[int]] = None,
    smooth_iterations: int = 0,
) -> vtk.vtkPolyData:
    """Extract polygonal outlines from a label map as ``vtkPolyData``.

    Args:
        lbl_img (vtk.vtkImageData):
            Image data where the scalars are integer label IDs.
        label_ids (list of int, optional):
            Specific label IDs to extract. Defaults to all IDs greater
            than 0 found in the data range.
        smooth_iterations (int, optional):
            Number of WindowedSinc smoothing iterations. A value of 0
            disables smoothing. Defaults to 0.

    Returns:
        vtk.vtkPolyData:
            A polydata object with a scalar array encoding label IDs
            per cell or point.
    """
    # Use vtkDiscreteMarchingCubes so each label produces its own surface
    dmc = vtk.vtkDiscreteMarchingCubes()
    dmc.SetInputData(lbl_img)

    # Determine which IDs to extract
    if label_ids is None:
        rng = lbl_img.GetPointData().GetScalars().GetRange()
        # conservative integer range; skip 0 (usually background)
        low = max(1, int(np.floor(rng[0])))
        high = int(np.ceil(rng[1]))
        ids = list(range(low, max(low, high) + 1))
    else:
        ids = list(map(int, label_ids))

    for k in ids:
        dmc.SetValue(k, float(k))

    dmc.Update()
    surf: vtk.vtkPolyData = dmc.GetOutput()

    if smooth_iterations > 0:
        ws = vtk.vtkWindowedSincPolyDataFilter()
        ws.SetInputData(surf)
        ws.SetNumberOfIterations(smooth_iterations)
        ws.BoundarySmoothingOff()
        ws.FeatureEdgeSmoothingOff()
        ws.NormalizeCoordinatesOn()
        ws.Update()
        surf = ws.GetOutput()

    return surf


def tiff_to_vtk(  # noqa: PLR0913, PLR0915, PLR0912, C901
    image_dir: str,
    output_path: str,
    channel_map: Dict[str, str],
    scaling_values: Union[List[float], Tuple[float, float, float]],
    label_dir: Optional[str] = None,
    ignore: Optional[List[str]] = ("Merge",),
    write_individual_vti: bool = False,
    write_label_surfaces: bool = False,
    surface_label_ids: Optional[List[int]] = None,
    surface_smooth_iterations: int = 0,
) -> str:
    """Convert a TIFF image set to a VTK multi-block bundle (.vtm).

    The output `.vtm` file contains:

    * One block per image channel as ``vtkImageData`` (Z, Y, X → VTK dims X, Y, Z).
    * Optional label maps as ``vtkImageData`` (integer IDs).
    * Optional outline surfaces (iso-surfaces) for labels as ``vtkPolyData``.

    Args:
        image_dir (str):
            Directory containing input TIFF images (z-stacks by channel).
        output_path (str):
            Path for the `.vtm` bundle; sibling `.vti` or `.vtp` files may
            also be written.
        channel_map (dict):
            Mapping from filename codes to human-readable channel names.
        scaling_values (tuple of float):
            Physical voxel spacing as ``(sz, sy, sx)`` in micrometers.
        label_dir (str, optional):
            Directory containing label TIFFs (2D or 3D). Two-dimensional
            labels are expanded or tiled along Z to match images.
        ignore (tuple of str, optional):
            Filename codes to ignore. Defaults to ``("Merge",)``.
        write_individual_vti (bool, optional):
            If ``True``, also write each channel/label as separate `.vti`
            files. Defaults to ``False``.
        write_label_surfaces (bool, optional):
            If ``True``, extract polygonal outlines (``vtkPolyData``) for
            label IDs and include them in the bundle. Also writes `.vtp` if
            ``write_individual_vti`` is ``True``. Defaults to ``False``.
        surface_label_ids (list of int, optional):
            Specific label IDs to extract. Defaults to all IDs greater than
            0 found in the data range.
        surface_smooth_iterations (int, optional):
            Number of smoothing iterations for surfaces. A value of 0
            disables smoothing. Defaults to 0.

    Returns:
        str: Path to the written `.vtm` file.
    """
    out = pathlib.Path(output_path)
    if out.exists():
        raise FileExistsError(
            f"Output path {output_path} already exists. Please remove it first."
        )
    out.parent.mkdir(parents=True, exist_ok=True)

    # Build image/label stacks using your existing code
    frame_zstacks = image_set_to_arrays(
        image_dir=image_dir,
        label_dir=label_dir,
        channel_map=channel_map,
        ignore=list(ignore) if ignore else None,
    )

    # Determine reference Z for labels that might be 2D
    first_img_stack = next(iter(frame_zstacks["images"].values()))
    z_ref = int(first_img_stack.shape[0])

    # Create a multiblock container
    bundle = vtk.vtkMultiBlockDataSet()
    block_names: List[str] = []

    # Helper to name blocks
    def set_block(idx: int, name: str, data_obj: vtk.vtkDataObject) -> None:
        bundle.SetBlock(idx, data_obj)
        meta = bundle.GetMetaData(idx)
        meta.Set(vtk.vtkCompositeDataSet.NAME(), name)

    # 1) Add image channels as vtkImageData
    idx = 0
    for channel_name, stack in frame_zstacks["images"].items():
        vtk_img = _np_to_vtk_image(
            arr=stack.astype(np.uint16, copy=False),
            spacing=(
                float(scaling_values[0]),
                float(scaling_values[1]),
                float(scaling_values[2]),
            ),
        )
        _add_string_field(vtk_img, "units", "micrometers")
        _add_string_field(vtk_img, "kind", "image")
        set_block(idx, f"image::{channel_name}", vtk_img)
        block_names.append(f"image::{channel_name}")

        if write_individual_vti:
            w = vtk.vtkXMLImageDataWriter()
            w.SetFileName(str(out.with_suffix(f".{channel_name}.vti")))
            w.SetInputData(vtk_img)
            if w.Write() != 1:
                raise IOError(
                    (f"Failed to write {out.with_suffix(f'.{channel_name}.vti')}")
                )
        idx += 1

    # 2) Add labels (if provided)
    if "labels" in frame_zstacks:
        for label_name, lbl_arr in frame_zstacks["labels"].items():
            # Ensure label map is 3D: (Z,Y,X)
            if lbl_arr.ndim == 2:  # noqa: PLR2004
                # Expand 2D to match Z: repeat along Z
                lbl_arr = np.repeat(lbl_arr[None, ...], z_ref, axis=0)  # noqa: PLW2901
            elif lbl_arr.ndim != 3:  # noqa: PLR2004
                raise ValueError(
                    (
                        "Label array must be 2D or 3D. "
                        f"Got shape {lbl_arr.shape} for {label_name}"
                    )
                )

            vtk_lbl = _np_to_vtk_image(
                arr=lbl_arr.astype(np.uint32, copy=False),  # allow many classes
                spacing=(
                    float(scaling_values[0]),
                    float(scaling_values[1]),
                    float(scaling_values[2]),
                ),
            )
            _add_string_field(vtk_lbl, "units", "micrometers")
            _add_string_field(vtk_lbl, "kind", "labels")
            set_block(idx, f"labels::{label_name}", vtk_lbl)
            block_names.append(f"labels::{label_name}")

            if write_individual_vti:
                w = vtk.vtkXMLImageDataWriter()
                w.SetFileName(str(out.with_suffix(f".{label_name}.labels.vti")))
                w.SetInputData(vtk_lbl)
                if w.Write() != 1:
                    raise IOError(
                        (
                            "Failed to write "
                            f"{out.with_suffix(f'.{label_name}.labels.vti')}"
                        )
                    )
            idx += 1

            # 3) Optional outlines as vtkPolyData
            if write_label_surfaces:
                surf = _discrete_surfaces_from_labels(
                    vtk_lbl,
                    label_ids=surface_label_ids,
                    smooth_iterations=surface_smooth_iterations,
                )
                _add_string_field(surf, "source", label_name)
                _add_string_field(surf, "kind", "label_surfaces")
                set_block(idx, f"surfaces::{label_name}", surf)
                block_names.append(f"surfaces::{label_name}")

                if write_individual_vti:
                    wp = vtk.vtkXMLPolyDataWriter()
                    wp.SetFileName(str(out.with_suffix(f".{label_name}.surfaces.vtp")))
                    wp.SetInputData(surf)
                    if wp.Write() != 1:
                        raise IOError(
                            (
                                "Failed to write "
                                f"{out.with_suffix(f'.{label_name}.surfaces.vtp')}"
                            )
                        )
                idx += 1

    # 4) Write the bundle (.vtm)
    wmb = vtk.vtkXMLMultiBlockDataWriter()
    wmb.SetFileName(str(out.with_suffix(".vtm")))
    wmb.SetInputData(bundle)
    if wmb.Write() != 1:
        raise IOError(f"Failed to write {out.with_suffix('.vtm')}")

    return str(out.with_suffix(".vtm"))
