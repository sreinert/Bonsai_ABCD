import numpy as np
from pathlib import Path

import suite2p
from suite2p.extraction.masks import create_cell_pix, create_neuropil_masks, create_masks, create_cell_mask
from suite2p.extraction.extract import extract_traces_from_masks
from suite2p.detection import roi_stats

sessions = ['TAA0000066/ses-023_date-20250516_protocol-t17'] 

for session in sessions: 
    print(f"Processing session: {session}")

    # Read in mask data from cellpose
    cellpose_fpath = "/ceph/mrsic_flogel/public/projects/AtApSuKuSaRe_20250129_HFScohort2/" + session + "/funcimg/Session/suite2p/plane0/meanImg_seg.npy"
    cellpose_masks = np.load(cellpose_fpath, allow_pickle=True).item()

    # Read in existing data from a suite2p run. We will use the "ops" and registered binary.
    # This should be the folder where processed suite2p files are saved. Consider making a backup copy of this folder before starting
    wd_path = '/ceph/mrsic_flogel/public/projects/AtApSuKuSaRe_20250129_HFScohort2/' + session + '/funcimg/Session/suite2p/plane0/'
    wd = Path(wd_path)
    ops = np.load(wd/'ops.npy', allow_pickle=True).item()
    Lx = ops['Lx']
    Ly = ops['Ly']
    f_reg = suite2p.io.BinaryFile(Ly, Lx, wd/'data.bin')


    # Using these inputs, we will first mimic the stat array made by suite2p
    masks = cellpose_masks['masks']
    stat = []
    for u_ix, u in enumerate(np.unique(masks)[1:]):
        ypix,xpix = np.nonzero(masks==u)
        npix = len(ypix)
        stat.append({'ypix': ypix, 'xpix': xpix, 'npix': npix, 'lam': np.ones(npix, np.float32), 'med': [np.mean(ypix), np.mean(xpix)]})
    stat = np.array(stat)
    stat = roi_stats(stat, Ly, Lx)  # This function fills in remaining roi properties to make it compatible with the rest of the suite2p pipeline/GUI

    # Using the constructed stat file, get masks
    cell_masks, neuropil_masks = create_masks(stat, Ly, Lx, ops)

    # Feed these values into the wrapper functions
    stat_after_extraction, F, Fneu, F_chan2, Fneu_chan2 = suite2p.extraction_wrapper(stat, f_reg, f_reg_chan2 = None,ops=ops)

    # Do cell classification
    classfile = suite2p.classification.builtin_classfile
    iscell = suite2p.classify(stat=stat_after_extraction, classfile=classfile)

    # Apply preprocessing step for deconvolution
    dF = F.copy() - ops['neucoeff']*Fneu
    dF = suite2p.extraction.preprocess(
            F=dF,
            baseline=ops['baseline'],
            win_baseline=ops['win_baseline'],
            sig_baseline=ops['sig_baseline'],
            fs=ops['fs'],
            prctile_baseline=ops['prctile_baseline']
        )
    # Identify spikes
    spks = suite2p.extraction.oasis(F=dF, batch_size=ops['batch_size'], tau=ops['tau'], fs=ops['fs'])


    # Overwrite files in wd folder (consider backing up this folder first)
    np.save(wd/'F.npy', F)
    np.save(wd/'Fneu.npy', Fneu)
    np.save(wd/'iscell.npy', iscell)
    np.save(wd/'ops.npy', ops)
    np.save(wd/'spks.npy', spks)
    np.save(wd/'stat.npy', stat)

    print(f"Finished updating session: {session}")
