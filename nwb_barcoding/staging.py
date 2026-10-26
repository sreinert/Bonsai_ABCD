"""Read-only staging helpers for NWB conversion inputs."""

from contextlib import contextmanager
from pathlib import Path
import tempfile

from . import compute_dff


def stage_raw_funcimg(raw_session, staging_root, scanimage_tif):
    """Create a minimal raw-session view containing one validated imaging acquisition."""
    raw_session = Path(raw_session)
    scanimage_tif = Path(scanimage_tif)
    source_funcimg = raw_session / "funcimg"
    if scanimage_tif.parent != source_funcimg:
        raise ValueError(f"TIFF {scanimage_tif} is not directly inside {source_funcimg}")
    vdaq_h5 = scanimage_tif.with_suffix(".h5")
    if not vdaq_h5.is_file():
        raise FileNotFoundError(f"Matching vDAQ recording does not exist: {vdaq_h5}")

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
