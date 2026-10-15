"""VRABCDConverter subclass that corrects behavior-side sample rates from the barcode fit.

See barcode_alignment.py's module docstring for the full diagnosis. In short:
Aimbot_barcoding's NIDAQDataInterFace and AeonDataInterFace assume fixed nominal
sample rates (9000.0 Hz and 9000/200 Hz respectively) when converting sample
index -> time, and only apply the barcode fit's intercept (a constant time
offset) on top of that. This subclass additionally corrects the *rate* itself
using the slope from the same barcode fit, which is what actually accounts for
the two DAQs' independent clocks drifting relative to each other over a
session. Nothing else about the conversion changes.
"""

from . import _aimbot_path
_aimbot_path.ensure_on_path()

from preprocess_alignment.nwb.behavior.VRABCDconverter import VRABCDConverter  # noqa: E402


class RobustVRABCDConverter(VRABCDConverter):
    """Same as VRABCDConverter, but also rescales NIDAQData/AeonPandasData's
    assumed sampling rate using the empirically-fit barcode slope instead of
    leaving it at the hardcoded nominal value.
    """

    def set_corrected_rates(self, nidaq_rate: float = None, aeon_rate: float = None):
        """Call this before `run_conversion(align_barcode=True, ...)`.

        Parameters
        ----------
        nidaq_rate : corrected NIDAQData.sampling_frequency (Hz), from
            barcode_alignment.fit_alignment()['corrected_secondary_rate'].
        aeon_rate : corrected AeonPandasData.sampling_frequency (Hz). If not
            given but nidaq_rate is, defaults to nidaq_rate/200 -- the same
            9000/200 ratio AeonDataInterFace's own hardcoded rate uses,
            since both are derived from the same underlying NIDAQ buffer clock.
        """
        self._nidaq_corrected_rate = nidaq_rate
        if aeon_rate is None and nidaq_rate is not None:
            aeon_rate = nidaq_rate / 200
        self._aeon_corrected_rate = aeon_rate

    def temporally_align_data_interfaces(self, align_barcode_deltat):
        super().temporally_align_data_interfaces(align_barcode_deltat)

        nidaq_rate = getattr(self, "_nidaq_corrected_rate", None)
        if nidaq_rate is not None and "NIDAQData" in self.data_interface_objects:
            interface = self.data_interface_objects["NIDAQData"]
            print(f"Correcting NIDAQData sampling_frequency: {interface.sampling_frequency} Hz "
                  f"(hardcoded nominal) -> {nidaq_rate:.4f} Hz (barcode-fit-derived)")
            interface.sampling_frequency = nidaq_rate

        aeon_rate = getattr(self, "_aeon_corrected_rate", None)
        if aeon_rate is not None and "AeonPandasData" in self.data_interface_objects:
            interface = self.data_interface_objects["AeonPandasData"]
            print(f"Correcting AeonPandasData sampling_frequency: {interface.sampling_frequency} Hz "
                  f"(hardcoded nominal) -> {aeon_rate:.4f} Hz (barcode-fit-derived)")
            interface.sampling_frequency = aeon_rate
