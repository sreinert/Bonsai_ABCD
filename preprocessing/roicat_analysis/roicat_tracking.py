from pathlib import Path
import multiprocessing as mp
import tempfile
import argparse
import matplotlib.pyplot as plt
import numpy as np
import roicat
import os

#%% Define inputs and basic variables
parser = argparse.ArgumentParser(description="Run ROICaT with animal and session inputs.")
parser.add_argument('--animal', type=str, default='TAA0000066', help="The animal ID (e.g. 'TAA0000066')")
parser.add_argument('--sessions', type=str, nargs='+', help="List of sessions to run ROICaT on")
args = parser.parse_args()

animal =  args.animal 
sessions = args.sessions

basepath = f'/media/mrsic_flogel/public/projects/AtApSuKuSaRe_20250129_HFScohort2/{animal}'
dir_save = os.path.join(basepath, 'ROICaT')
Path(dir_save).mkdir(parents=True, exist_ok=True)
Path(os.path.join(dir_save, 'visualization')).mkdir(parents=True, exist_ok=True)

DEVICE = roicat.helpers.set_device(use_GPU=True) # Set the device to run on 
SEED = roicat.util.set_random_seed(seed=None, deterministic=False)  # Set determinism and seed

#%% Data input
# Find paths to data 
dir_allOuterFolders = basepath

pathSuffixToStat = 'stat.npy'
pathSuffixToOps = 'ops.npy'

paths_allStat = [str(Path(dir_allOuterFolders) / session / 'funcimg/Session/suite2p/plane0' / pathSuffixToStat) for session in sessions]
paths_allOps  = [str(Path(path).resolve().parent / pathSuffixToOps) for path in paths_allStat]

print(f'paths to all stat files:')
[print(path) for path in paths_allStat]
print('')
print(f'paths to all ops files:')
[print(path) for path in paths_allOps]

# Import data 
data = roicat.data_importing.Data_suite2p(
    paths_statFiles=paths_allStat[:],
    paths_opsFiles=paths_allOps[:],
    um_per_pixel=1.3523,  ## IMPORTANT PARAMETER. Use a list of floats if values differ in each session.
    new_or_old_suite2p='new',
    type_meanImg='meanImgE',
    verbose=True,
)

assert data.check_completeness(verbose=False)['tracking'], f"Data object is missing attributes necessary for tracking."

# Display imported images
roicat.helpers.save_gif(
    array=roicat.helpers.add_text_to_images(
        images=[((f / np.max(f)) * 255).astype(np.uint8) for f in data.FOV_images], 
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
        images=[((f / np.max(f)) * 255).astype(np.uint8) for f in data.get_maxIntensityProjection_spatialFootprints()], 
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
        images=[((f / np.max(f)) * 255).astype(np.uint8) for f in FOV_images], 
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
        images=[((f / np.max(f)) * 255).astype(np.uint8) for f in aligner.ims_registered_geo], 
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
        images=[((f / np.max(f)) * 255).astype(np.uint8) for f in aligner.ims_registered_nonrigid], 
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
        images=[((f / np.max(f)) * 255).astype(np.uint8) for f in aligner.get_ROIsAligned_maxIntensityProjection(normalize=True)], 
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
        images=[((f / np.max(f)) * 255).astype(np.uint8) for f in blurrer.get_ROIsBlurred_maxIntensityProjection()], 
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
dir_temp = tempfile.gettempdir()

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
    numWorkers_dataloader=mp.cpu_count(),  ## (advanced) PyTorch dataloader num_workers
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
sim = roicat.tracking.similarity_graph.ROI_graph(
    n_workers=-1,  ## Number of CPU cores to use. -1 for all.
    frame_height=data.FOV_height,
    frame_width=data.FOV_width,
    block_height=128,  ## size of a block
    block_width=128,  ## size of a block
    algorithm_nearestNeigbors_spatialFootprints='brute',  ## algorithm used to find the pairwise similarity for s_sf. ('brute' is slow but exact. See docs for others.)
    verbose=True,  ## Whether to print outputs
)

fig=sim.visualize_blocks()
fig.savefig(str(Path(dir_save).resolve() / 'visualization' / 'similarity_graph.png'))

s_sf, s_NN, s_SWT, s_sesh = sim.compute_similarity_blockwise(
    spatialFootprints=blurrer.ROIs_blurred,  ## Mask spatial footprints
    features_NN=roinet.latents,  ## ROInet output latents
    features_SWT=swt.latents,  ## Scattering wavelet transform output latents
    ROI_session_bool=data.session_bool,  ## Boolean array of which ROIs belong to which sessions
    spatialFootprint_maskPower=1.0,  ##  An exponent to raise the spatial footprints to to care more or less about bright pixels
)

sim.make_normalized_similarities(
    centers_of_mass=data.centroids,  ## ROI centroid positions
    features_NN=roinet.latents,  ## ROInet latents
    features_SWT=swt.latents,  ## SWT latents
    k_max=data.n_sessions*100,  ## Maximum number of nearest neighbors to consider for the normalizing distribution
    k_min=data.n_sessions*10,  ## Minimum number of nearest neighbors to consider for the normalizing distribution
    algo_NN='kd_tree',  ## Nearest neighbors algorithm to use
    device=DEVICE,
)

#%% Clustering

# 1. Mix the similarity matrices into a single distance matrix
# Initialize the clusterer object by passing the similarity matrices in
clusterer = roicat.tracking.clustering.Clusterer(
    s_sf=sim.s_sf,
    s_NN_z=sim.s_NN_z,
    s_SWT_z=sim.s_SWT_z,
    s_sesh=sim.s_sesh,
    verbose=1,
)

# Automatically find mixing parameters
kwargs_makeConjunctiveDistanceMatrix_best = clusterer.find_optimal_parameters_for_pruning(
    n_bins=None,  ## Number of bins to use for the histograms of the distributions. If None, then a heuristic is used.
    smoothing_window_bins=None,  ## Number of bins to use to smooth the distributions. If None, then a heuristic is used.
    kwargs_findParameters={
        'n_patience': 300,  ## Number of optimization epoch to wait for tol_frac to converge
        'tol_frac': 0.001,  ## Fractional change below which optimization will conclude
        'max_trials': 1200,  ## Max number of optimization epochs
        'max_duration': 60*10,  ## Max amount of time (in seconds) to allow optimization to proceed for
        'value_stop': 0.0,  ## Goal value. If value equals or goes below value_stop, optimization is stopped.
    },
    bounds_findParameters={
        'power_NN': [0.0, 2.],  ## Bounds for the exponent applied to s_NN
        'power_SWT': [0.0, 2.],  ## Bounds for the exponent applied to s_SWT
        'p_norm': [-5, -0.1],  ## Bounds for the p-norm p value (Minkowski) applied to mix the matrices
        'sig_NN_kwargs_mu': [0., 1.0],  ## Bounds for the sigmoid center for s_NN
        'sig_NN_kwargs_b': [0.1, 1.5],  ## Bounds for the sigmoid slope for s_NN
        'sig_SWT_kwargs_mu': [0., 1.0],  ## Bounds for the sigmoid center for s_SWT
        'sig_SWT_kwargs_b': [0.1, 1.5],  ## Bounds for the sigmoid slope for s_SWT
    },
    n_jobs_findParameters=-1,  ## Number of CPU cores to use (-1 is all cores)
    seed=SEED,  ## Random seed
)

fig = clusterer.plot_distSame(kwargs_makeConjunctiveDistanceMatrix=kwargs_makeConjunctiveDistanceMatrix_best)
fig.savefig(str(Path(dir_save).resolve() / 'visualization' / 'distance_matrix_sameClusters.png'))

fig, _ = clusterer.plot_similarity_relationships(
    plots_to_show=[1,2,3], 
    max_samples=100000,  ## Make smaller if it is running too slow
    kwargs_scatter={'s':1, 'alpha':0.2},
    kwargs_makeConjunctiveDistanceMatrix=kwargs_makeConjunctiveDistanceMatrix_best
)
fig.savefig(str(Path(dir_save).resolve() / 'visualization' / 'distance_matrix_diffClusters.png'))

# 2. Prune the distance matrix 
clusterer.make_pruned_similarity_graphs(
    d_cutoff=None,  ## Optionally manually specify a distance cutoff
    kwargs_makeConjunctiveDistanceMatrix=kwargs_makeConjunctiveDistanceMatrix_best,
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
        thresh_cost=0.8,  ## Threshold. Higher values result in more permissive clustering. Specifically, the pairwise metric distance between ROIs above which two ROIs cannot be clustered together.
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
fig, _ = roicat.tracking.clustering.plot_quality_metrics(quality_metrics=quality_metrics, labels=labels_squeezed, n_sessions=data.n_sessions);
fig.savefig(str(Path(dir_save).resolve() / 'visualization' / 'quality_metrics.png'))

# Color visualisation of the results 
FOV_clusters = roicat.visualization.compute_colored_FOV(
    spatialFootprints=[r.power(1.0) for r in results_all['ROIs']['ROIs_aligned']],  ## Spatial footprint sparse arrays
    FOV_height=results_all['ROIs']['frame_height'],
    FOV_width=results_all['ROIs']['frame_width'],
    labels=results_all["clusters"]["labels_bySession"],  ## cluster labels
)

roicat.helpers.save_gif(
    array=roicat.helpers.add_text_to_images(
        images=[((f / np.max(f)) * 255).astype(np.uint8) for f in FOV_clusters], 
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

