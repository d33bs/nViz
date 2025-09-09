"""
Utilities for viewing n-dimensional data
"""

import logging
from typing import Dict, Iterable, Optional

import napari
import numpy as np
import pyvista as pv
import tifffile as tiff
import vtk
import xmltodict
import zarr

logger = logging.getLogger(__name__)


def view_zarr_with_napari(
    zarr_dir: str, scaling_values: tuple, headless: bool = False
) -> Optional[napari.Viewer]:
    """
    View a Zarr file created with nviz through napari.

    Args:
        zarr_dir (str):
            The path to the Zarr file.
        scaling_values (tuple):
            The scaling values for the image.
        headless (bool):
            Whether to run in headless mode
            (where we don't run napari and hand
            back the viewer object instead).

    Returns:
        Optional[napari.Viewer]]:
            The napari viewer object if not headless,
            otherwise None.
    """
    # Check Zarr file structure
    frame_zarr = zarr.open(zarr_dir, mode="r")

    # Visualize with napari, start in 3d mode
    viewer = napari.Viewer(ndisplay=3)

    # Iterate through each channel in the Zarr file
    for channel_name in sorted(frame_zarr["images"].keys(), reverse=True):
        viewer.add_image(
            frame_zarr["images"][channel_name]["0"][:],
            name=channel_name,
            scale=scaling_values,
        )

    # Iterate through each compartment in the Zarr file and add labels to Napari
    if "labels" in frame_zarr:
        for label_name in sorted(frame_zarr["labels"].keys(), reverse=True):
            viewer.add_labels(
                frame_zarr["labels"][label_name]["0"][:],
                name=f"{label_name}",
                scale=scaling_values,
            )

    if not headless:
        # Start the Napari event loop
        napari.run()
    else:
        logger.warning(
            "Running view in headless mode and returning a napari viewer object."
        )

    # otherwise return the viewer
    return viewer


def view_ometiff_with_napari(
    ometiff_path: str, scaling_values: tuple, headless: bool = False
) -> Optional[napari.Viewer]:
    """
    View a OME-TIFF file created with nviz through napari.

    Args:
        ometiff_path (str):
            The path to the OME-TIFF file.
        scaling_values (tuple):
            The scaling values for the image.
        headless (bool):
            Whether to run in headless mode
            (where we don't run napari and hand
            back the viewer object instead).

    Returns:
        Optional[napari.Viewer]]:
            The napari viewer object if not headless,
            otherwise None.
    """

    # Visualize with napari, start in 3d mode
    viewer = napari.Viewer(ndisplay=3)

    # Read and add layers from the combined OME-TIFF file
    with tiff.TiffFile(ometiff_path) as tif:
        combined_data = tif.asarray()
        metadata = xmltodict.parse(tif.ome_metadata)
        channel_names = [
            channel["@Name"]
            for channel in metadata["OME"]["Image"]["Pixels"]["Channel"]
        ]

        # First, add image layers
        for i, (channel_data, channel_name) in enumerate(
            zip(combined_data, channel_names)
        ):
            if "(labels)" not in channel_name:
                viewer.add_image(
                    channel_data,
                    name=channel_name,
                    scale=scaling_values,
                )

        # Then, add label layers
        for i, (channel_data, channel_name) in enumerate(
            zip(combined_data, channel_names)
        ):
            if "(labels)" in channel_name:
                viewer.add_labels(
                    channel_data,
                    name=channel_name,
                    scale=scaling_values,
                )

    if not headless:
        # Start the Napari event loop
        napari.run()
    else:
        logger.warning(
            "Running view in headless mode and returning a napari viewer object."
        )

    # otherwise return the viewer
    return viewer


def _ensure_pyvista_jupyter_backend() -> None:
    """Use the trame-backed browser renderer inside Jupyter."""
    try:
        pv.set_jupyter_backend("trame")
    except Exception as exc:  # pragma: no cover
        logger.warning("Falling back to default PyVista backend: %s", exc)


def _build_label_lut(max_id: int, alpha: float = 0.5) -> pv.LookupTable:
    """Create a categorical LUT (0=transparent background)."""
    lut = pv.LookupTable(n_values=max(2, int(max_id) + 1))
    lut.scalar_range = (0, max(1, int(max_id)))
    # Set background (0) fully transparent
    lut.SetTableValue(0, 0.0, 0.0, 0.0, 0.0)

    # Simple, reproducible palette for IDs >= 1
    rng = np.random.default_rng(42)
    for k in range(1, int(max_id) + 1):
        r, g, b = rng.random(3)
        lut.SetTableValue(k, float(r), float(g), float(b), float(alpha))
    return lut


def _is_image(obj: vtk.vtkObject) -> bool:
    return isinstance(obj, vtk.vtkImageData)


def _is_poly(obj: vtk.vtkObject) -> bool:
    return isinstance(obj, vtk.vtkPolyData)


def _block_name(bundle: vtk.vtkMultiBlockDataSet, idx: int) -> str:
    meta = bundle.GetMetaData(idx)
    key = vtk.vtkCompositeDataSet.NAME()
    return meta.Get(key) if meta and meta.Has(key) else f"block_{idx}"


def view_vtm_with_pyvista(  # noqa: PLR0913, C901, PLR0912
    vtm_path: str,
    *,
    opacity: "str|float|Iterable[float]" = "sigmoid",
    label_alpha: float = 0.40,
    show_axes: bool = True,
    lighting: bool = True,
    bg_color: str = "black",
    jupyter: bool = True,
    headless: bool = False,
    screenshot_path: Optional[str] = None,
    gif_path: Optional[str] = None,
    gif_frames: int = 120,
) -> Optional[pv.Plotter]:
    """
    View a VTK multiblock bundle (.vtm) with PyVista in a notebook-friendly way.

    The bundle may contain:
      - image::<channel>         (vtkImageData, grayscale/intensity)
      - labels::<name>           (vtkImageData, integer IDs)
      - surfaces::<name>         (vtkPolyData, outlines/meshes)

    Args:
        vtm_path:
            Path to a `.vtm` file (from the exporter).
        opacity:
            Volume opacity. Use a string ("linear", "sigmoid", "geom"), a float,
            or an iterable transfer curve.
        label_alpha:
            Base alpha to apply to label LUT entries (ID>=1).
        show_axes:
            Show a small axes widget.
        lighting:
            Enable shading for volumes.
        bg_color:
            Plotter background color.
        jupyter:
            Configure PyVista for interactive Jupyter rendering (trame).
        headless:
            If True, do not call `.show()`. Return the plotter for the caller.
        screenshot_path:
            If provided, save a PNG after rendering.
        gif_path:
            If provided, save a rotating GIF (camera azimuth).
        gif_frames:
            Number of frames for the GIF rotation.

    Returns:
        Optionally the `pyvista.Plotter` (when `headless=True`), else None.
    """
    if jupyter and not headless:
        _ensure_pyvista_jupyter_backend()

    # Read the multiblock bundle
    reader = vtk.vtkXMLMultiBlockDataReader()
    reader.SetFileName(vtm_path)
    reader.Update()
    bundle: vtk.vtkMultiBlockDataSet = reader.GetOutput()

    offscreen = bool(screenshot_path or gif_path) or headless
    pl = pv.Plotter(off_screen=offscreen)
    pl.set_background(bg_color)

    # First pass: add intensity volumes and collect label stats
    label_max_by_block: Dict[int, int] = {}

    for i in range(bundle.GetNumberOfBlocks()):
        obj = bundle.GetBlock(i)
        name = _block_name(bundle, i)

        if _is_image(obj):
            # Heuristic: treat blocks named "labels::" as label volumes
            if name.startswith("labels::"):
                # record max ID for LUT sizing
                scalars = obj.GetPointData().GetScalars()
                if scalars is not None:
                    rng = scalars.GetRange()
                    label_max_by_block[i] = int(rng[1])
                continue  # add labels in second pass
            # regular image channel
            grid = pv.wrap(obj)
            pl.add_volume(
                grid,
                opacity=opacity,
                shade=bool(lighting),
                cmap="gray",
                name=name,
            )

    # Second pass: add label volumes with discrete LUTs
    for i, max_id in label_max_by_block.items():
        obj = bundle.GetBlock(i)
        name = _block_name(bundle, i)
        grid = pv.wrap(obj)

        lut = _build_label_lut(max_id=max_id, alpha=label_alpha)
        pl.add_volume(
            grid,
            opacity=1.0,  # opacity is controlled by LUT alpha
            cmap=lut,  # categorical, no interpolation
            shade=False,
            name=name,
        )

    # Third pass: add any outline surfaces
    for i in range(bundle.GetNumberOfBlocks()):
        obj = bundle.GetBlock(i)
        name = _block_name(bundle, i)
        if _is_poly(obj):
            pl.add_mesh(pv.wrap(obj), opacity=0.5, name=name)

    if show_axes:
        pl.add_axes()

    # Headless caller wants to decide when/how to render
    if headless:
        logger.warning(
            "Running VTK viewer in headless mode; returning the PyVista Plotter."
        )
        return pl

    # Interactive display (Jupyter/Lab)
    pl.show(
        jupyter_backend="trame" if pv.global_theme.jupyter_backend == "trame" else None
    )

    # Optional captures (works even if interactive window already closed)
    if screenshot_path:
        pl.screenshot(filename=screenshot_path)

    if gif_path:
        pl.open_gif(gif_path)
        for _ in range(gif_frames):
            pl.camera.azimuth(360.0 / gif_frames)
            pl.render()
            pl.write_frame()
        pl.close()  # closes animation writer (not the plotter window)

    return None


def view_vti_with_pyvista(  # noqa: PLR0913
    vti_path: str,
    *,
    opacity: "str|float|Iterable[float]" = "sigmoid",
    lighting: bool = True,
    bg_color: str = "black",
    jupyter: bool = True,
    headless: bool = False,
    screenshot_path: Optional[str] = None,
) -> Optional[pv.Plotter]:
    """
    Convenience viewer for a single `.vti` image (vtkImageData).

    Useful when you exported individual channels/labels as separate files.
    """
    if jupyter and not headless:
        _ensure_pyvista_jupyter_backend()

    r = vtk.vtkXMLImageDataReader()
    r.SetFileName(vti_path)
    r.Update()
    img = r.GetOutput()

    pl = pv.Plotter(off_screen=bool(screenshot_path))
    pl.set_background(bg_color)
    pl.add_volume(pv.wrap(img), opacity=opacity, shade=bool(lighting))
    if not headless:
        pl.show(
            jupyter_backend="trame"
            if pv.global_theme.jupyter_backend == "trame"
            else None
        )
    if screenshot_path:
        pl.screenshot(filename=screenshot_path)
    return pl if headless else None
