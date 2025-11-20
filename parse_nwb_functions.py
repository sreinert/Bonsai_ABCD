from pathlib import Path
import yaml
from pynwb import NWBHDF5IO
from typing import Literal
from pynwb import NWBHDF5IO

def find_base_path(mouse,session,root):
    base_path = None
    mouse_path = Path(root) / f"sub-{mouse}" 

    for folder in mouse_path.iterdir():
        if folder.is_dir() and session in folder.name:
            print(f"Found folder: {folder}")
            base_path = folder
    return base_path

def load_settings(base_path):
    settings_path = Path(base_path) / "behav"
    settings_file = list(settings_path.glob("*.yaml"))[0]
    with open(settings_file, "r") as f:
        ses_settings = yaml.safe_load(f)

    return ses_settings, None

def find_nwbfile(session_path: Path, format: Literal["merged", "behav", "funcimg"] = "merged"):
    """
    Find NWB file
    """
    if format == "merged":
        fname = list(session_path.glob('*.nwb'))[0]
    elif format == "behav":
        fname = list((session_path / 'behav').glob('*.nwb'))[0]
    elif format == "funcimg":
        fname = list((session_path / 'funcimg').glob('*.nwb'))[0]

    return fname

def get_ttl_onsets(nwb_path, event_name=None):

    io = NWBHDF5IO(nwb_path, mode='r')
    nwb = io.read()

    NIDAQ_TTLTypesTable = nwb.acquisition['NIDAQ_TTLTypesTable'][:]
    print('TTL onsets recorded by NIDAQ:')
    print(NIDAQ_TTLTypesTable)

    if event_name is not None:
        print(f'Retrieve {event_name}')
        matches = NIDAQ_TTLTypesTable[NIDAQ_TTLTypesTable["event_name"] == event_name].index.tolist()
        if len(matches) == 1:
            NIDAQ_TTLsTable = nwb.acquisition['NIDAQ_TTLsTable'][:]
            mask = NIDAQ_TTLsTable["ttl_type"].isin(matches)
            return NIDAQ_TTLsTable.loc[mask, "timestamp"].values
        else:
            raise ValueError('Check event_name')

def get_s2pstat_nwb(nwb_path):
    io = NWBHDF5IO(nwb_path, mode='r')
    nwb = io.read()

    mean_img = nwb.processing['ophys'].data_interfaces['SegmentationImages'].images['MeanImageChan1Plane0'][:]

    segmentation = nwb.processing['ophys'].data_interfaces['ImageSegmentation'].plane_segmentations['PlaneSegmentationChan1Plane0'][:]
    image_mask = segmentation['image_mask']
    ROICentroids = segmentation['ROICentroids']

    return mean_img, image_mask, ROICentroids