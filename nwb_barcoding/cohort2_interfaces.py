"""neuroconv interfaces + converter for cohort2's (pre-Bonsai) behaviour data.

Cohort2 counterparts of Aimbot_barcoding's NIDAQDataInterFace/AeonDataInterFace/VRABCDConverter,
which can't be reused as-is: NIDAQDataInterFace reads Bonsai rig-settings JSON to find its
channels and parses the session time from a Bonsai-style folder name, and AeonDataInterFace
reads a Bonsai/Aeon event dataframe. Cohort2 instead has a bare interleaved *_ai.bin and a
VR position_log.csv on its own clock (see cohort2_layout.py).

What IS reused unchanged: VRABCDConverter's run_conversion (in particular its
append-into-an-existing-NWB branch, which is how the merged NWB is built) and the
NIDAQ_TTLTypesTable/NIDAQ_TTLsTable/SyncTTL naming, so the resulting NWBs have the same
structure as the cohort3 ones and verify_session.py's generic checks apply to both.

Alignment is applied the same way RobustVRABCDConverter does for cohort3 -- slope + offset
from barcode_alignment.fit_alignment(), not offset only -- but expressed directly as
`t_aligned = t_nidaq_nominal * slope + delta_t` on every timestamp.
"""

import datetime as dt
from pathlib import Path

import numpy as np
from neuroconv import BaseDataInterface
from pynwb import NWBFile, TimeSeries
from ndx_events import TtlsTable, TtlTypesTable

from . import _aimbot_path
_aimbot_path.ensure_on_path()

from preprocess_alignment.nwb.behavior.VRABCDconverter import VRABCDConverter  # noqa: E402

from . import cohort2_layout as layout  # noqa: E402


class Cohort2NIDAQInterface(BaseDataInterface):
    """NIDAQ *_ai.bin -> NIDAQ_TTLTypesTable/NIDAQ_TTLsTable (+ SyncTTL TimeSeries when unaligned)."""

    display_name = "Cohort2 NIDAQ analog data"
    info = "Interleaved float64 NIDAQ analog recording from the pre-Bonsai VR rig (cohort2)."

    def __init__(self, ai_file_path: str, n_channels: int, session_start_time: str,
                 ttl_channels_to_names_map: dict, sync_channel: int = 0,
                 threshold_v: float = layout.TTL_THRESHOLD_V, verbose: bool = False):
        """
        :param ai_file_path: path to *_ai.bin
        :param n_channels: number of interleaved channels (see cohort2_layout.ai_channel_count)
        :param session_start_time: ISO datetime string
        :param ttl_channels_to_names_map: {"<column index>": "<event name>"} for event channels
        :param sync_channel: column holding the barcode, written as the SyncTTL TimeSeries
        """
        super().__init__(ai_file_path=ai_file_path, n_channels=n_channels,
                         session_start_time=session_start_time,
                         ttl_channels_to_names_map=ttl_channels_to_names_map,
                         sync_channel=sync_channel, threshold_v=threshold_v, verbose=verbose)
        self.ai = layout.load_ai(Path(ai_file_path), n_channels)
        self.sampling_frequency = layout.NIDAQ_NOMINAL_RATE
        self.slope, self.delta_t, self.aligned = 1.0, 0.0, False

    def set_alignment(self, slope: float, delta_t: float):
        self.slope, self.delta_t, self.aligned = slope, delta_t, True

    def get_metadata(self) -> dict:
        metadata = super().get_metadata()
        metadata["NWBFile"].update(
            session_start_time=dt.datetime.fromisoformat(self.source_data["session_start_time"]))
        metadata["Events"] = dict(
            NIDAQ_TTLTypesTable=dict(name="NIDAQ_TTLTypesTable",
                                     description="Contains the type of TTL signals from NIDAQ output."),
            NIDAQ_TTLsTable=dict(name="NIDAQ_TTLsTable",
                                 description="Contains the TTL signals onset times."),
        )
        return metadata

    def get_event_times(self, column: int) -> np.ndarray:
        """Rising-edge times (s) of one analog column, on the aligned clock if set."""
        edges = layout.detect_rising_edges(self.ai[:, column], threshold=self.source_data["threshold_v"])
        # +1: the edge index is the last sample below threshold; the event is the next sample.
        return (edges + 1) / self.sampling_frequency * self.slope + self.delta_t

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata: dict) -> None:
        events_metadata = metadata["Events"]
        ttl_types_table = TtlTypesTable(**events_metadata["NIDAQ_TTLTypesTable"])
        ttls_table = TtlsTable(**events_metadata["NIDAQ_TTLsTable"], target_tables={"ttl_type": ttl_types_table})

        channel_map = {int(k): v for k, v in self.source_data["ttl_channels_to_names_map"].items()}
        for ttl_type, (column, name) in enumerate(channel_map.items()):
            ttl_types_table.add_row(event_name=str(name),
                                    event_type_description=f"The onset times of the {name} event "
                                                           f"(NIDAQ ai column {column}).",
                                    pulse_value=np.uint8(1), check_ragged=False)
            for timestamp in self.get_event_times(column):
                ttls_table.add_row(ttl_type=ttl_type, timestamp=timestamp, check_ragged=False)

        nwbfile.add_acquisition(ttl_types_table)
        nwbfile.add_acquisition(ttls_table)

        # The merged NWB is a copy of the funcimg NWB, which already holds the vDAQ 'SyncTTL';
        # like cohort3's align_barcode=True branch, the NIDAQ barcode trace only goes into the
        # unaligned behav NWB (that's where barcode_alignment.read_syncttl reads it from).
        if not self.aligned:
            sync = self.ai[:, self.source_data["sync_channel"]] >= self.source_data["threshold_v"]
            nwbfile.add_acquisition(TimeSeries(
                name="SyncTTL", data=sync.astype(np.uint8), starting_time=0.0,
                rate=self.sampling_frequency, unit="Volts",
                description=f"Barcode TTL (NIDAQ ai column {self.source_data['sync_channel']}), "
                            f"digitized at {self.source_data['threshold_v']} V."))


class Cohort2VRLogInterface(BaseDataInterface):
    """VR position_log.csv -> Position/TotalRunDistance TimeSeries + one TimeSeries per VR event type.

    VR-log times are first mapped onto the NIDAQ clock with the reward-based fit from
    cohort2_layout.fit_vr_to_nidaq(), then onto the aligned clock like everything else.
    Values are written as logged (Position == -1 means "no position", as in the raw log).
    """

    display_name = "Cohort2 VR log"
    info = "position_log.csv from the pre-Bonsai VR (cohort2)."

    def __init__(self, file_path: str, vr_to_nidaq_slope: float, vr_to_nidaq_intercept: float,
                 verbose: bool = False):
        super().__init__(file_path=file_path, vr_to_nidaq_slope=vr_to_nidaq_slope,
                         vr_to_nidaq_intercept=vr_to_nidaq_intercept, verbose=verbose)
        self.vr_log = layout.load_vr_log(file_path)
        self.slope, self.delta_t = 1.0, 0.0

    def set_alignment(self, slope: float, delta_t: float):
        self.slope, self.delta_t = slope, delta_t

    def get_timestamps(self) -> np.ndarray:
        nidaq_t = (self.vr_log['Time'].values * self.source_data["vr_to_nidaq_slope"]
                   + self.source_data["vr_to_nidaq_intercept"])
        return nidaq_t * self.slope + self.delta_t

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata: dict) -> None:
        timestamps = self.get_timestamps()
        for column, unit in [('Position', 'cm'), ('TotalRunDistance', 'cm')]:
            nwbfile.add_acquisition(TimeSeries(
                name=column, data=self.vr_log[column].to_numpy(dtype=float), timestamps=timestamps,
                unit=unit, description=f"VR-logged {column} (-1 = not logged), VR clock mapped onto "
                                       f"the NIDAQ clock via the reward-matched fit."))

        events = self.vr_log['Event']
        for event_name in sorted(events.dropna().unique()):
            mask = (events == event_name).values
            safe = ''.join(c if c.isalnum() else '_' for c in str(event_name))
            nwbfile.add_acquisition(TimeSeries(
                name=f"VR_event_{safe}", data=np.ones(int(mask.sum())), timestamps=timestamps[mask],
                unit="n.a.", description=f"Times of VR-logged '{event_name}' events."))


class Cohort2VRABCDConverter(VRABCDConverter):
    """VRABCDConverter wired to the cohort2 interfaces.

    Reuses VRABCDConverter.get_metadata/run_conversion unchanged; only the interfaces and
    the alignment step differ. Call set_alignment_slope() before
    run_conversion(align_barcode=True, align_barcode_deltat=delta_t, ...).
    """

    data_interface_classes = dict(
        NIDAQData=Cohort2NIDAQInterface,
        VRLogData=Cohort2VRLogInterface,
    )

    def set_alignment_slope(self, slope: float):
        self._slope = slope

    def temporally_align_data_interfaces(self, align_barcode_deltat):
        slope = getattr(self, "_slope", None)
        if slope is None:
            raise RuntimeError("set_alignment_slope() must be called before an aligned conversion -- "
                               "an offset-only alignment is exactly the bug barcode_alignment.py fixes.")
        print(f"Aligning behaviour: t_aligned = t_nidaq * {slope:.9f} + {align_barcode_deltat:.6f} s")
        for interface in self.data_interface_objects.values():
            interface.set_alignment(slope=slope, delta_t=align_barcode_deltat)
