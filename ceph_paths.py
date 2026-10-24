"""Resolve lab-share (ceph) paths on whichever machine the code is running on.

The same mrsic_flogel share is mounted in different places:

    Mac (local)    /Volumes/mrsic_flogel/...
    SWC HPC        /ceph/mrsic_flogel/...
    Windows rigs   Y:/...  or  Z:/...   (drive letter differs per machine)

Paths used to be hardcoded with one of these prefixes, so notebooks and scripts only ran on
the machine they were written on. This module detects the mount once at import time and
translates any of the forms above into the local one:

    from ceph_paths import ceph_path, CEPH_ROOT, PROJECTS_ROOT

    root = ceph_path('/Volumes/mrsic_flogel/public/projects/SuKuSaRe_20250923_HFScohort3/rawdata')
    # -> '/ceph/mrsic_flogel/public/projects/SuKuSaRe_20250923_HFScohort3/rawdata' on the HPC

Existing hardcoded paths therefore only need wrapping in ceph_path(...), and can keep
whichever prefix they were written with. Paths not on the share (e.g. ~/Desktop/...) are
returned unchanged.

Detection order:
  1. the CEPH_ROOT environment variable, if set (e.g. a share mounted somewhere unusual);
  2. the first mount that exists: /ceph/mrsic_flogel, /Volumes/mrsic_flogel, Y:/, Z:/;
  3. if none exists (share not mounted), the usual mount for this OS, so paths still build
     and the error appears where a file is actually opened, as it did before.

The share is not accessed beyond those existence checks, so importing this module is cheap
and works even when the share is down.
"""

import os
import re
import sys
from pathlib import Path, PurePath

HPC_ROOT = '/ceph/mrsic_flogel'
MAC_ROOT = '/Volumes/mrsic_flogel'
WINDOWS_ROOTS = ('Y:', 'Z:')

# Any known share prefix, with / or \ separators, at the start of a path.
_SHARE_PREFIX = re.compile(r'^(?:/ceph/mrsic_flogel|/Volumes/mrsic_flogel|[YZ]:)(?=[/\\]|$)', re.IGNORECASE)


def _detect_root():
    env_root = os.environ.get('CEPH_ROOT')
    if env_root:
        return env_root.rstrip('/\\')
    for root in (HPC_ROOT, MAC_ROOT) + WINDOWS_ROOTS:
        if os.path.isdir(root + '/'):
            return root
    if sys.platform == 'darwin':
        return MAC_ROOT
    if sys.platform.startswith('win'):
        return WINDOWS_ROOTS[0]
    return HPC_ROOT


CEPH_ROOT = _detect_root()
ON_HPC = CEPH_ROOT == HPC_ROOT
PROJECTS_ROOT = f'{CEPH_ROOT}/public/projects'


def ceph_path(path):
    """Translate a share path written for any machine into this machine's mount.

    Accepts str or Path and returns the same type. Separators after the share prefix are
    normalised to '/', and a trailing separator is kept (some code concatenates with +).
    Non-share paths come back unchanged.
    """
    as_path = isinstance(path, PurePath)
    s = str(path)
    m = _SHARE_PREFIX.match(s)
    if m is None:
        return path
    rest = s[m.end():].replace('\\', '/')
    out = CEPH_ROOT + re.sub(r'/+', '/', rest)
    return Path(out) if as_path else out


if __name__ == '__main__':
    print(f'CEPH_ROOT     = {CEPH_ROOT}')
    print(f'PROJECTS_ROOT = {PROJECTS_ROOT}')
