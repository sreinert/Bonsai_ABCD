"""Read-only staging helpers for NWB conversion inputs."""

from contextlib import contextmanager
from pathlib import Path
import tempfile

from . import compute_dff


TIFF_SUFFIXES = {".tif", ".tiff"}


def _visible_tiffs(funcimg):
    return sorted(
        path for path in Path(funcimg).iterdir()
        if path.is_file()
        and not path.name.startswith("._")
        and path.suffix.lower() in TIFF_SUFFIXES
    )


def select_scanimage_tif(raw_session, suite2p_plane_path):
    """Select the TIFF recorded by Suite2p, including a cropped derivative.

    A raw directory may legitimately contain both ``movie.tif`` and
    ``movie_cropped.tif``.  In that case Suite2p's ``ops.npy`` is the source of
    truth; guessing would risk combining traces from one movie with vDAQ timing
    from another acquisition.
    """
    funcimg = Path(raw_session) / "funcimg"
    tiffs = _visible_tiffs(funcimg)
    if not tiffs:
        raise FileNotFoundError(f"No ScanImage TIFF found in {funcimg}")
    if len(tiffs) == 1:
        return tiffs[0]

    import numpy as np

    ops_path = Path(suite2p_plane_path) / "ops.npy"
    ops = np.load(ops_path, allow_pickle=True).item()
    files_processed = ops.get("file_list", ops.get("filelist"))
    if files_processed is None:
        raise RuntimeError(
            f"Found multiple ScanImage TIFFs in {funcimg}, but {ops_path} does not "
            "record Suite2p's input file list. Cannot safely choose between "
            f"{[path.name for path in tiffs]}."
        )
    processed = [
        Path(value.decode() if isinstance(value, bytes) else str(value))
        for value in files_processed
    ]
    if len(processed) != 1:
        raise RuntimeError(
            f"Suite2p processed {len(processed)} TIFFs ({[str(path) for path in processed]}); "
            "the NWB pipeline requires exactly one acquisition."
        )

    processed_path = processed[0]
    exact_matches = [path for path in tiffs if path.resolve() == processed_path.resolve()]
    if len(exact_matches) == 1:
        return exact_matches[0]
    name_matches = [path for path in tiffs if path.name == processed_path.name]
    if len(name_matches) == 1:
        return name_matches[0]
    raise RuntimeError(
        f"Suite2p says it processed {processed_path}, but that TIFF is not one of "
        f"{[path.name for path in tiffs]} in {funcimg}."
    )


def resolve_vdaq_h5(scanimage_tif):
    """Resolve vDAQ data, preferring an explicitly recovered recording."""
    scanimage_tif = Path(scanimage_tif)
    source_stem = scanimage_tif.stem
    if source_stem.lower().endswith("_cropped"):
        source_stem = source_stem[:-len("_cropped")]

    recovered = sorted(
        scanimage_tif.parent.glob(f"{source_stem}_recovered_syncG*_upto_*.h5")
    )
    if len(recovered) == 1:
        return recovered[0]
    if len(recovered) > 1:
        raise RuntimeError(
            f"Multiple recovered vDAQ recordings match {scanimage_tif}: "
            f"{[path.name for path in recovered]}. Keep only the intended recovery "
            "or select it explicitly in the pipeline."
        )

    candidates = [
        scanimage_tif.with_suffix(".h5"),
        scanimage_tif.with_name(source_stem + ".h5"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Matching vDAQ recording does not exist for {scanimage_tif}; tried "
        f"{[str(path) for path in candidates]}"
    )


def resolve_scanimage_metadata_tif(scanimage_tif):
    """Use the uncropped source TIFF when a crop has lost ScanImage metadata.

    The NWB conversion does not embed the raw movie, but Aimbot still opens one
    TIFF to read its ScanImage header. Rewritten recovery TIFFs commonly lose
    that private metadata, while their source TIFF remains the correct metadata
    companion for the Suite2p traces.
    """
    scanimage_tif = Path(scanimage_tif)
    if not scanimage_tif.stem.lower().endswith("_cropped"):
        return scanimage_tif

    source_stem = scanimage_tif.stem[:-len("_cropped")]
    candidates = [
        scanimage_tif.with_name(source_stem + scanimage_tif.suffix),
        *(
            scanimage_tif.with_name(source_stem + suffix)
            for suffix in sorted(TIFF_SUFFIXES.difference({scanimage_tif.suffix.lower()}))
        ),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"The cropped Suite2p input {scanimage_tif} has no uncropped TIFF available "
        "for ScanImage metadata; tried "
        f"{[str(path) for path in candidates]}"
    )


def stage_raw_funcimg(raw_session, staging_root, scanimage_tif, vdaq_h5=None):
    """Create a minimal raw-session view containing one validated imaging acquisition."""
    raw_session = Path(raw_session)
    scanimage_tif = Path(scanimage_tif)
    source_funcimg = raw_session / "funcimg"
    if scanimage_tif.parent != source_funcimg:
        raise ValueError(f"TIFF {scanimage_tif} is not directly inside {source_funcimg}")
    vdaq_h5 = Path(vdaq_h5) if vdaq_h5 is not None else resolve_vdaq_h5(scanimage_tif)
    if vdaq_h5.parent != source_funcimg or not vdaq_h5.is_file():
        raise FileNotFoundError(f"Resolved vDAQ recording is invalid: {vdaq_h5}")

    raw_view = Path(staging_root) / "rawdata"
    staged_funcimg = raw_view / "funcimg"
    staged_funcimg.mkdir(parents=True)
    for source in (scanimage_tif, vdaq_h5):
        (staged_funcimg / source.name).symlink_to(source)

    # Preserve optional Aimbot inputs when present; ignore AppleDouble sidecars.
    for pattern in ("experiment*.h5", "phase_mask*.csv"):
        for source in sorted(source_funcimg.glob(pattern)):
            if source.is_file() and not source.name.startswith("._"):
                (staged_funcimg / source.name).symlink_to(source)
    return raw_view


@contextmanager
def staged_tracking_layout(suite2p_dir, output_session, source_session=None,
                           dff_neuropil_coefficient=None, force=False):
    """Provide Aimbot's expected ``funcimg/{suite2p,dff}`` layout without touching inputs."""
    suite2p_dir = Path(suite2p_dir).expanduser().resolve()
    output_session = Path(output_session)
    if not suite2p_dir.is_dir():
        raise FileNotFoundError(f"Suite2p directory does not exist: {suite2p_dir}")

    output_session.mkdir(parents=True, exist_ok=True)
    # Keep derived staging data on the local temporary filesystem. Network shares can
    # create delayed AppleDouble files while a directory is being removed, causing
    # TemporaryDirectory cleanup to fail even after conversion succeeded.
    with tempfile.TemporaryDirectory(prefix="nwb-barcoding-stage-") as tmp:
        tracking_root = Path(tmp) / "tracking"
        staged_funcimg = tracking_root / "funcimg"
        staged_funcimg.mkdir(parents=True)
        (staged_funcimg / "suite2p").symlink_to(suite2p_dir, target_is_directory=True)

        if source_session is not None:
            source_anat = Path(source_session) / "anat"
            if source_anat.is_dir():
                (tracking_root / "anat").symlink_to(source_anat, target_is_directory=True)

        compute_dff.ensure_dff(
            funcimg_dir=staged_funcimg,
            neuropil_coefficient=dff_neuropil_coefficient,
            force=force,
        )
        yield tracking_root
