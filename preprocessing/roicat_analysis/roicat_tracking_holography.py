"""ROICaT cross-session ROI tracking for the holography cohort (SaReSuKuMaRa_20260911_holography/cohort1).

Copy of roicat_tracking.py (written for cohort2, TAA00000xx mice). The original is left untouched.
The tracking pipeline (alignment, blurring, ROInet + SWT embedding, similarity graph, clustering,
outputs) is unchanged, so explore_roicat_output.ipynb / roi_tracking_helpers.py read the results the
same way. What changed:

- Layout: single-tree cohort. suite2p output is read from
  <root>/preprocessed/<animal>/ses-<tag>_*/funcimg/suite2p/plane0/{stat,ops}.npy, the same folder
  nwb_barcoding/build_session_nwb.py uses. (A few sessions also have an older funcimg/func_suite2p/
  run; it is ignored.) Sessions are given as tags (2LM001 ... 2LM004 = t0 ... t3) and resolved by glob,
  requiring exactly one match.
- Output goes to <root>/analysis/<animal>/ROICaT, never into rawdata/ or preprocessed/.
- um_per_pixel = 0.834: ScanImage imagingFovUm is +-213.498 um at 512 px for all 16 holography
  sessions (426.996 / 512). cohort2's 1.3523 would mis-scale ROInet inputs and alignment radii.
- Dataloader workers = CPUs allocated to the job (os.sched_getaffinity), not mp.cpu_count(), which on a
  cluster node counts every core on the node.
- Agg backend for headless GPU nodes; sessions_used.json records which folder each session index
  (= label order in the results) came from.
- --dry_run only resolves and prints the input paths.
- roicat >= 1.7 API (hpc/pyproject.toml pins 1.7.12): the original was written for an older roicat and
  crashes at ROI_graph(algorithm_nearestNeigbors_spatialFootprints=...). The similarity graph and
  Clusterer now take metric_configs=DEFAULT_METRICS and features/similarities as dicts; the mixing
  parameter search (naive-Bayes sigmoid calibration + differential evolution) runs with roicat defaults
  because the old optuna kwargs/bounds no longer exist; sequential-Hungarian thresh_cost=None (= pruning
  d_cutoff) instead of the old fixed 0.8. Calls follow roicat.pipelines.pipeline_tracking.
- ROInet model is downloaded to $ROICAT_MODEL_DIR (default /tmp/roicat_models_<user>), not the shared /tmp.
- Diagnostic figures go through save_fig(), so a plotting failure can't kill a run after clustering;
  GIF frames are NaN-safe (non-rigid warping leaves NaN borders, which made them black).

Run (see hpc/README.md for the SLURM array + uv env):
    python roicat_tracking_holography.py --animal sub-001 --sessions 2LM001 2LM002 2LM003 2LM004
"""
from pathlib import Path
import multiprocessing as mp
import tempfile
import argparse
import json
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))  # repo root, for ceph_paths
from ceph_paths import ceph_path

#%% Define inputs and basic variables
parser = argparse.ArgumentParser(description="Run ROICaT on holography-cohort sessions of one animal.")
parser.add_argument('--animal', type=str, required=True, help="Subject folder, e.g. 'sub-001'")
parser.add_argument('--sessions', type=str, nargs='+', required=True,
                    help="Session tags in order, e.g. 2LM001 2LM002 2LM003 2LM004")
parser.add_argument('--root', type=str,
                    default=ceph_path('/ceph/mrsic_flogel/public/projects/SaReSuKuMaRa_20260911_holography/cohort1'),
                    help="Cohort folder containing preprocessed/ and analysis/")
parser.add_argument('--um_per_pixel', type=float, default=426.996 / 512,
                    help="Imaging resolution (default from ScanImage imagingFovUm, 512 px)")
parser.add_argument('--dry_run', action='store_true', help="Only resolve and print the input paths")
args = parser.parse_args()

animal = args.animal
sessions = args.sessions
root = Path(ceph_path(args.root))

# Resolve each session tag to its preprocessed folder
session_dirs = []
for tag in sessions:
    matches = sorted((root / 'preprocessed' / animal).glob(f'ses-{tag}_*'))
    if len(matches) != 1:
        sys.exit(f'Expected exactly one session folder for {animal} {tag}, found {len(matches)}: {matches}')
    session_dirs.append(matches[0])

paths_allStat = [str(d / 'funcimg/suite2p/plane0/stat.npy') for d in session_dirs]
paths_allOps  = [str(d / 'funcimg/suite2p/plane0/ops.npy') for d in session_dirs]
missing = [p for p in paths_allStat + paths_allOps if not Path(p).exists()]
if missing:
    sys.exit(f'Missing suite2p files: {missing}')

print(f'paths to all stat files:')
[print(path) for path in paths_allStat]
print('')
print(f'paths to all ops files:')
[print(path) for path in paths_allOps]
print(f'um_per_pixel: {args.um_per_pixel}')

if args.dry_run:
    sys.exit(0)

import roicat

dir_save = str(root / 'analysis' / animal / 'ROICaT')
Path(dir_save).mkdir(parents=True, exist_ok=True)
Path(os.path.join(dir_save, 'visualization')).mkdir(parents=True, exist_ok=True)

def to_uint8(f):
    """Normalise an image to uint8; non-rigid warping leaves NaNs at the FOV borders."""
    f = np.nan_to_num(np.asarray(f, dtype=float), nan=0.0)
    return ((f / max(f.max(), 1e-12)) * 255).astype(np.uint8)

def save_fig(make_fig, name):
    """Save a diagnostic figure without letting a plotting failure kill the run."""
    try:
        fig = make_fig()
        if fig is None:
            print(f'{name}: nothing to plot')
            return
        fig.savefig(str(Path(dir_save).resolve() / 'visualization' / name))
        plt.close(fig)
    except Exception as e:
        print(f'{name}: plotting failed ({type(e).__name__}: {e})')

def save_current_fig(draw, name):
    """For roicat plot methods that draw into a new figure but return None."""
    save_fig(lambda: (draw(), plt.gcf())[1], name)

with open(os.path.join(dir_save, 'sessions_used.json'), 'w') as f:
    json.dump([{'index': i, 'tag': tag, 'session_dir': str(d), 'stat': s, 'ops': o}
               for i, (tag, d, s, o) in enumerate(zip(sessions, session_dirs, paths_allStat, paths_allOps))],
              f, indent=2)

try:
    n_cpus = len(os.sched_getaffinity(0))  # CPUs allocated to this job (Linux)
except AttributeError:
    n_cpus = mp.cpu_count()  # macOS

DEVICE = roicat.helpers.set_device(use_GPU=True) # Set the device to run on
SEED = roicat.util.set_random_seed(seed=None, deterministic=False)  # Set determinism and seed

#%% Data input
# Import data 
data = roicat.data_importing.Data_suite2p(
    paths_statFiles=paths_allStat[:],
    paths_opsFiles=paths_allOps[:],
    um_per_pixel=args.um_per_pixel,  ## IMPORTANT PARAMETER. Use a list of floats if values differ in each session.
    new_or_old_suite2p='new',
    type_meanImg='meanImgE',
    verbose=True,
)

assert data.check_completeness(verbose=False)['tracking'], f"Data object is missing attributes necessary for tracking."

# Display imported images
roicat.helpers.save_gif(
    array=roicat.helpers.add_text_to_images(
        images=[to_uint8(f) for f in data.FOV_images], 
        text=[[f"{ii}",] for ii in range(len(data.FOV_images))], 
        font_size=3,
        line_width=10,
        position=(30, 90),
    ), 
    path=str(Path(dir_save).resolve() / 'visualization' / 'input_FOV_images.gif'),
    frameRate=10,
    loop=0,
)

roicat.helpers.save_gif(
    array=roicat.helpers.add_text_to_images(
        images=[to_uint8(f) for f in data.get_maxIntensityProjection_spatialFootprints()], 
        text=[[f"{ii}",] for ii in range(len(data.FOV_images))], 
        font_size=3,
        line_width=10,
        position=(30, 90),
    ), 
    path=str(Path(dir_save).resolve() / 'visualization' / 'maxIntensityProjection_spatialFootprints_input_FOV_images.gif'),
    frameRate=10,
    loop=0,
)

#%% Alignment 
aligner = roicat.tracking.alignment.Aligner(
    use_match_search=True,  ## Use our algorithm for doing all-pairs matching if template matching fails.
    all_to_all=False,  ## Force the use of our algorithm for all-pairs matching. Much slower (False: O(N) vs. True: O(N^2)), but more accurate.
    radius_in=4,  ## IMPORTANT PARAMETER: Value in micrometers used to define the maximum shift/offset between two images that are considered to be aligned. Larger means more lenient alignment.
    radius_out=20,  ## Value in micrometers used to define the minimum shift/offset between two images that are considered to be misaligned.
    z_threshold=4.0,  ## IMPORTANT PARAMETER: Z-score required to define two images as aligned. Larger values results in more stringent alignment requirements.
    um_per_pixel=data.um_per_pixel[0],  ## Single value for um_per_pixel. data.um_per_pixel is typically a list of floats, so index out just one value.
    device=DEVICE,
    verbose=True,
)

# 1. FOV augmentation 
FOV_images = aligner.augment_FOV_images(
    FOV_images=data.FOV_images,
    spatialFootprints=data.spatialFootprints,
    normalize_FOV_intensities=True,
    roi_FOV_mixing_factor=0.5,
    use_CLAHE=True,  ## IMPORTANT PARAMETER. Use Set to False if data is poor quality or poorly aligned.
    CLAHE_grid_block_size=10,  ## IMPORTANT PARAMETER. Use smaller values for higher precision but higher chance of failure.
    CLAHE_clipLimit=1.0,
    CLAHE_normalize=True,
)

# Display augmented images
roicat.helpers.save_gif(
    array=roicat.helpers.add_text_to_images(
        images=[to_uint8(f) for f in FOV_images], 
        text=[[f"{ii}",] for ii in range(len(FOV_images))], 
        font_size=3,
        line_width=10,
        position=(30, 90),
    ), 
    path=str(Path(dir_save).resolve() / 'visualization' / 'augmented_FOV_images.gif'),
    frameRate=10,
    loop=0,
)

# 2. Fit geometric transformation 
aligner.fit_geometric(
    template=0.5,  ## specifies which image to use as the template. Either array (image), integer (ims_moving index), or float (ims_moving fractional index)
    ims_moving=FOV_images,  ## input images
    template_method='sequential',  ## 'sequential': align images to neighboring images (good for drifting data). 'image': align to a single image
    mask_borders=(0, 0, 0, 0),  ## number of pixels to mask off the edges (top, bottom, left, right)
    method='DISK_LightGlue',  ## See below for options.
    kwargs_method = {
        'RoMa': {  ## Accuracy: Best, Speed: Very slow (can be fast with a GPU).
            'model_type': 'outdoor',
            'n_points': 10000,  ## Higher values mean more points are used for the registration. Useful for larger FOV_images. Larger means slower.
            'batch_size': 1000,
        },
        'DISK_LightGlue': {  ## Accuracy: Good, Speed: Fast.
            'num_features': 3000,  ## Number of features to extract and match. I've seen best results around 2048 despite higher values typically being better.
            'threshold_confidence': 0.0,  ## Higher values means fewer but better matches.
            'window_nms': 7,  ## Non-maximum suppression window size. Larger values mean fewer non-suppressed points.
        },
        'LoFTR': {  ## Accuracy: Okay. Speed: Medium.
            'model_type': 'indoor_new',
            'threshold_confidence': 0.2,  ## Higher values means fewer but better matches.
        },
        'ECC_cv2': {  ## Accuracy: Okay. Speed: Medium.
            'mode_transform': 'euclidean',  ## Must be one of {'translation', 'affine', 'euclidean', 'homography'}. See cv2 documentation on findTransformECC for more details.
            'n_iter': 200,
            'termination_eps': 1e-09,  ## Termination criteria for the registration algorithm. See documentation for more details.
            'gaussFiltSize': 1,  ## Size of the gaussian filter used to smooth the FOV_image before registration. Larger values mean more smoothing.
            'auto_fix_gaussFilt_step': 10,  ## If the registration fails, then the gaussian filter size is reduced by this amount and the registration is tried again.
        },
        'PhaseCorrelation': {  ## Accuracy: Poor. Speed: Very fast. Notes: Only applicable for translations, not rotations or scaling.
            'bandpass_freqs': [1, 30],
            'order': 5,
        },
    },
    constraint='affine',  ## Must be one of {'rigid', 'euclidean', 'similarity', 'affine', 'homography'}. Choose constraint based on expected changes in images; use the simplest constraint that is applicable.
    kwargs_RANSAC = {
        'inl_thresh': 3.0,  ## cv2.findHomography RANSAC inlier threshold. Larger values mean more lenient matching.
        'max_iter': 100,
        'confidence': 0.99,
    },
    verbose=True,  ## Set to 3 to view plots of the alignment process if available for the method.
)

fig, _ = aligner.plot_alignment_results_geometric()
fig.savefig(str(Path(dir_save).resolve() / 'visualization' / 'FOV_images_aligned_geometric.png'))

# 3. Fit non-rigid transformation 
aligner.fit_nonrigid(
    template=0.5,  ## specifies which image to use as the template. Either array (image), integer (ims_moving index), or float (ims_moving fractional index)
    ims_moving=aligner.ims_registered_geo,  ## Input images. Typically the geometrically registered images
    remappingIdx_init=aligner.remappingIdx_geo,  ## The remappingIdx between the original images (and ROIs) and ims_moving
    template_method='image',  ## 'sequential': align images to neighboring images. 'image': align to a single image, good if using geometric registration first
    method='DeepFlow',
    kwargs_method = {
        'DeepFlow': {},  ## Accuracy: Good (good in middle, poor on edges), Speed: Fast (CPU only)
        'RoMa': {  ## Accuracy: Okay (decent in middle, poor on edges), Speed: Slow (can be fast with a GPU), Notes: This method can work on the raw images without pre-registering using geometric methods.
            'model_type': 'outdoor',
        },
        'OpticalFlowFarneback': {  ## Accuracy: Varies (can sometimes be tuned to be the best as there are no edge artifacts), Speed: Medium (CPU only)
            'pyr_scale': 0.7,
            'levels': 5,
            'winsize': 256,
            'iterations': 15,
            'poly_n': 5,
            'poly_sigma': 1.5,            
        },
    },    
)

aligner.transform_images_nonrigid(FOV_images)
fig, _ = aligner.plot_alignment_results_nonrigid()
fig.savefig(str(Path(dir_save).resolve() / 'visualization' / 'FOV_images_aligned_non-rigid.png'))

# 4. Transform ROIs
aligner.transform_ROIs(
    ROIs=data.spatialFootprints, 
    remappingIdx=aligner.remappingIdx_nonrigid,
    # remappingIdx=aligner.remappingIdx_geo,
    normalize=True,
)

# Ensure that the aligned images look aligned 
roicat.helpers.save_gif(
    array=roicat.helpers.add_text_to_images(
        images=[to_uint8(f) for f in aligner.ims_registered_geo], 
        text=[[f"{ii}",] for ii in range(len(data.FOV_images))], 
        font_size=3,
        line_width=10,
        position=(30, 90),
    ), 
    path=str(Path(dir_save).resolve() / 'visualization' / 'FOV_images_aligned_geometric.gif'),
    frameRate=10,
    loop=0,
)

roicat.helpers.save_gif(
    array=roicat.helpers.add_text_to_images(
        images=[to_uint8(f) for f in aligner.ims_registered_nonrigid], 
        text=[[f"{ii}",] for ii in range(len(data.FOV_images))], 
        font_size=3,
        line_width=10,
        position=(30, 90),
    ), 
    path=str(Path(dir_save).resolve() / 'visualization' / 'FOV_images_aligned_non-rigid.gif'),
    frameRate=10,
    loop=0,
)

roicat.helpers.save_gif(
    array=roicat.helpers.add_text_to_images(
        images=[to_uint8(f) for f in aligner.get_ROIsAligned_maxIntensityProjection(normalize=True)], 
        text=[[f"{ii}",] for ii in range(len(data.FOV_images))], 
        font_size=3,
        line_width=10,
        position=(30, 90),
    ), 
    path=str(Path(dir_save).resolve() / 'visualization' / 'maxIntensityProjection_spatialFootprints_aligned_FOV_images.gif'),
    frameRate=10,
    loop=0,
)

#%% Blur ROIs
blurrer = roicat.tracking.blurring.ROI_Blurrer(
    frame_shape=(data.FOV_height, data.FOV_width),  ## FOV height and width
    kernel_halfWidth=4,  ## The half width of the 2D gaussian used to blur the ROI masks
    plot_kernel=False,  ## Whether to visualize the 2D gaussian
)

blurrer.blur_ROIs(
    spatialFootprints=aligner.ROIs_aligned[:],
)

roicat.helpers.save_gif(
    array=roicat.helpers.add_text_to_images(
        images=[to_uint8(f) for f in blurrer.get_ROIsBlurred_maxIntensityProjection()], 
        text=[[f"{ii}",] for ii in range(len(data.FOV_images))], 
        font_size=3,
        line_width=10,
        position=(30, 90),
    ), 
    path=str(Path(dir_save).resolve() / 'visualization' / 'maxIntensityProjection_ROIsBlurred.gif'),
    frameRate=10,
    loop=0,
)

#%% ROInet embedding
# Per-user model dir: /tmp on the GPU nodes is shared, and another user's extracted ROInet files there
# can't be overwritten (PermissionError). The job sets ROICAT_MODEL_DIR to a persistent folder on ceph.
import getpass
dir_temp = os.environ.get('ROICAT_MODEL_DIR') or os.path.join(tempfile.gettempdir(), f'roicat_models_{getpass.getuser()}')
Path(dir_temp).mkdir(parents=True, exist_ok=True)

roinet = roicat.ROInet.ROInet_embedder(
    device=DEVICE,  ## Which torch device to use ('cpu', 'cuda', etc.)
    dir_networkFiles=dir_temp,  ## Directory to download the pretrained network to
    download_method='check_local_first',  ## Check to see if a model has already been downloaded to the location (will skip if hash matches)
    download_url='https://osf.io/x3fd2/download',  ## URL of the model
    download_hash='7a5fb8ad94b110037785a46b9463ea94',  ## Hash of the model file
    forward_pass_version='latent',  ## How the data is passed through the network
    verbose=True,  ## Whether to print updates
)

roinet.generate_dataloader(
    ROI_images=data.ROI_images,  ## Input images of ROIs
    um_per_pixel=data.um_per_pixel,  ## Resolution of FOV
    pref_plot=False,  ## Whether or not to plot the ROI sizes
    
    jit_script_transforms=False,  ## (advanced) Whether or not to use torch.jit.script to speed things up
    
    batchSize_dataloader=8,  ## (advanced) PyTorch dataloader batch_size
    pinMemory_dataloader=True,  ## (advanced) PyTorch dataloader pin_memory
    numWorkers_dataloader=n_cpus,  ## (advanced) PyTorch dataloader num_workers
    persistentWorkers_dataloader=True,  ## (advanced) PyTorch dataloader persistent_workers
    prefetchFactor_dataloader=2,  ## (advanced) PyTorch dataloader prefetch_factor
)

# roicat.visualization.display_toggle_image_stack(roinet.ROI_images_rs[:1000], image_size=(200,200))
roinet.generate_latents()

# Scattering wavelet embedding
swt = roicat.tracking.scatteringWaveletTransformer.SWT(
    kwargs_Scattering2D={'J': 2, 'L': 12},  ## 'J' is the number of convolutional layers. 'L' is the number of wavelet angles.
    image_shape=data.ROI_images[0].shape[1:3],  ## size of a cropped ROI image
    device=DEVICE,  ## PyTorch device
)

swt.transform(
    ROI_images=roinet.ROI_images_rs,  ## All the cropped and resized ROI images
    batch_size=100,  ## Batch size for each iteration (smaller is less memory but slower)
)

#%% Compute similarities
# roicat >= 1.7 API: metrics are configured via DEFAULT_METRICS ('sf', 'nn', 'swt') and features are
# passed as a dict; calls follow roicat.pipelines.pipeline_tracking.
from roicat.tracking.similarity_graph import DEFAULT_METRICS

sim = roicat.tracking.similarity_graph.ROI_graph(
    n_workers=-1,  ## Number of CPU cores to use. -1 for all.
    frame_height=data.FOV_height,
    frame_width=data.FOV_width,
    block_height=128,  ## size of a block
    block_width=128,  ## size of a block
    verbose=True,  ## Whether to print outputs
    metric_configs=DEFAULT_METRICS,
)

save_current_fig(sim.visualize_blocks, 'similarity_graph.png')  ## draws into a new figure, returns None

sim.compute_similarity_blockwise(
    spatialFootprints=blurrer.ROIs_blurred,  ## Mask spatial footprints
    ROI_session_bool=data.session_bool,  ## Boolean array of which ROIs belong to which sessions
    features={'nn': roinet.latents, 'swt': swt.latents},  ## ROInet and scattering wavelet latents
    spatialFootprint_maskPower=1.0,  ##  An exponent to raise the spatial footprints to to care more or less about bright pixels
)

sim.make_normalized_similarities(
    centers_of_mass=data.centroids,  ## ROI centroid positions
    features={'nn': roinet.latents, 'swt': swt.latents},  ## Feature vectors for z-scoring
    k_max=data.n_sessions*100,  ## Maximum number of nearest neighbors to consider for the normalizing distribution
    k_min=data.n_sessions*10,  ## Minimum number of nearest neighbors to consider for the normalizing distribution
    algo_NN='kd_tree',  ## Nearest neighbors algorithm to use
    device=DEVICE,
)

#%% Clustering

# 1. Mix the similarity matrices into a single distance matrix
clusterer = roicat.tracking.clustering.Clusterer(
    similarities=sim.similarities_final,
    metric_configs=DEFAULT_METRICS,
    s_sesh=sim.s_sesh,
    session_bool=data.session_bool,
    verbose=True,
)

# Automatically find mixing parameters (1.7: naive-Bayes calibration of the sigmoids + differential
# evolution over powers/p_norm; the old optuna kwargs/bounds no longer exist, so roicat defaults are used)
kwargs_makeConjunctiveDistanceMatrix_best = clusterer.find_optimal_parameters_for_pruning(
    seed=SEED,  ## Random seed
)

save_fig(lambda: clusterer.plot_distSame(mixing_params=kwargs_makeConjunctiveDistanceMatrix_best),
         'distance_matrix_sameClusters.png')
save_fig(lambda: clusterer.plot_similarity_relationships(
             max_samples=100000,  ## Make smaller if it is running too slow
             kwargs_scatter={'s':1, 'alpha':0.2},
             mixing_params=kwargs_makeConjunctiveDistanceMatrix_best,
         )[0],
         'distance_matrix_diffClusters.png')

# 2. Prune the distance matrix
clusterer.make_pruned_similarity_graphs(
    d_cutoff=None,  ## Optionally manually specify a distance cutoff
    mixing_params=kwargs_makeConjunctiveDistanceMatrix_best,
    stringency=1.0,  ## Modifies the threshold for pruning the distance matrix. Higher values result in LESS pruning. New d_cutoff = stringency * truncated d_cutoff.
    convert_to_probability=False,
)

# 3. Cluster
if data.n_sessions >= 6:
    labels = clusterer.fit(
        d_conj=clusterer.dConj_pruned,  ## Input distance matrix
        session_bool=data.session_bool,  ## Boolean array of which ROIs belong to which sessions
        min_cluster_size=2,  ## Minimum number of ROIs that can be considered a 'cluster'
        n_iter_violationCorrection=6,  ## Number of times to redo clustering sweep after removing violations
        split_intraSession_clusters=True,  ## Whether or not to split clusters with ROIs from the same session
        cluster_selection_method='leaf',  ## (advanced) Method of cluster selection for HDBSCAN (see hdbscan documentation)
        d_clusterMerge=None,  ## Distance below which all ROIs are merged into a cluster
        alpha=0.999,  ## (advanced) Scalar applied to distance matrix in HDBSCAN (see hdbscan documentation)
        discard_failed_pruning=True,  ## (advanced) Whether or not to set all ROIs that could be separated from clusters with ROIs from the same sessions to label=-1
        n_steps_clusterSplit=100,  ## (advanced) How finely to step through distances to remove violations
    )

else:
    labels = clusterer.fit_sequentialHungarian(
        d_conj=clusterer.dConj_pruned,  ## Input distance matrix
        session_bool=data.session_bool,  ## Boolean array of which ROIs belong to which sessions
        thresh_cost=None,  ## Pairwise distance above which two ROIs cannot be clustered together. None = d_cutoff from pruning (roicat 1.7 default; the old fixed 0.8 was on the pre-1.7 distance scale).
    )

# 4. Compute quality metrics
quality_metrics = clusterer.compute_quality_metrics()


#%% Collect results 
labels_squeezed, labels_bySession, labels_bool, labels_bool_bySession, labels_dict = roicat.tracking.clustering.make_label_variants(labels=labels, n_roi_bySession=data.n_roi)

results_clusters = {
    'labels': labels_squeezed,
    'labels_bySession': labels_bySession,
    'labels_dict': labels_dict,
    'quality_metrics': clusterer.quality_metrics if hasattr(clusterer, 'quality_metrics') else None,
}

results_all = {
    "clusters":{
        "labels": roicat.util.JSON_List(labels_squeezed),
        "labels_bySession": roicat.util.JSON_List(labels_bySession),
        "labels_bool": labels_bool,
        "labels_bool_bySession": labels_bool_bySession,
        "labels_dict": roicat.util.JSON_Dict(labels_dict),
        "quality_metrics": roicat.util.JSON_Dict(clusterer.quality_metrics) if hasattr(clusterer, 'quality_metrics') else None,
    },
    "ROIs": {
        "ROIs_aligned": aligner.ROIs_aligned,
        "ROIs_raw": data.spatialFootprints,
        "frame_height": data.FOV_height,
        "frame_width": data.FOV_width,
        "idx_roi_session": np.where(data.session_bool)[1],
        "n_sessions": data.n_sessions,
    },
    "input_data": {
        "paths_stat": data.paths_stat,
        "paths_ops": data.paths_ops,
    },
}

run_data = {
    'data': data.__dict__,
    'aligner': aligner.__dict__,
    'blurrer': blurrer.__dict__,
    'roinet': roinet.__dict__,
    'swt': swt.__dict__,
    'sim': sim.__dict__,
    'clusterer': clusterer.__dict__,
}

params_used = {name: mod['params'] for name, mod in run_data.items()}

# Visualize results 
print(f'Number of clusters: {len(np.unique(results_clusters["labels"]))}')
print(f'Number of discarded ROIs: {(np.array(results_clusters["labels"])==-1).sum()}')

# Plot quality metrics 
save_fig(lambda: roicat.tracking.clustering.plot_quality_metrics(quality_metrics=quality_metrics, labels_bySession=labels_bySession)[0],
         'quality_metrics.png')

# Color visualisation of the results 
FOV_clusters = roicat.visualization.compute_colored_FOV(
    spatialFootprints=[r.power(1.0) for r in results_all['ROIs']['ROIs_aligned']],  ## Spatial footprint sparse arrays
    FOV_height=results_all['ROIs']['frame_height'],
    FOV_width=results_all['ROIs']['frame_width'],
    labels=results_all["clusters"]["labels_bySession"],  ## cluster labels
)

roicat.helpers.save_gif(
    array=roicat.helpers.add_text_to_images(
        images=[to_uint8(f) for f in FOV_clusters], 
        text=[[f"{ii}",] for ii in range(len(FOV_clusters))], 
        font_size=3,
        line_width=10,
        position=(30, 90),
    ), 
    path=str(Path(dir_save).resolve() / 'visualization' / 'FOV_clusters.gif'),
    frameRate=10,
    loop=0,
)

# Save results
paths_save = {
    'results_clusters': str(Path(dir_save) / f'{animal}.tracking.results_clusters.json'),
    'params_used':      str(Path(dir_save) / f'{animal}.tracking.params_used.json'),
    'results_all':      str(Path(dir_save) / f'{animal}.tracking.results_all.richfile'),
    'run_data':         str(Path(dir_save) / f'{animal}.tracking.run_data.richfile'),
}

Path(dir_save).mkdir(parents=True, exist_ok=True)

roicat.helpers.json_save(obj=results_clusters, filepath=paths_save['results_clusters'])
roicat.helpers.json_save(obj=params_used, filepath=paths_save['params_used'])
roicat.util.RichFile_ROICaT(path=paths_save['results_all']).save(obj=results_all, overwrite=True)
roicat.util.RichFile_ROICaT(path=paths_save['run_data']).save(obj=run_data, overwrite=True)

