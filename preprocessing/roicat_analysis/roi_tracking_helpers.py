from pathlib import Path
import re, os, sys
import numpy as np
import matplotlib.pyplot as plt
import roicat
import scipy.sparse
import importlib
importlib.reload(roicat)

from IPython.display import HTML, display
import numpy as np
import base64
from PIL import Image
from io import BytesIO
import torch
import datetime
import hashlib


def load_neural_data(basepath, animal, sessions_to_align, data_type='F'):
    # Load neural data 
    data = [[] for s in range(len(sessions_to_align))]
    for s, sess in enumerate(sessions_to_align):
        print(f'Loading neural data for session {sess}')
        if data_type == 'F':
            datapath = basepath / animal / sess / 'funcimg' / 'Session' / 'suite2p' / 'plane0' / 'F.npy'
        elif data_type == 'DF_F0':
            datapath = basepath / animal / sess / 'funcimg' / 'Session' / 'suite2p' / 'plane0' / 'DF_F0.npy'
            if not os.path.exists(datapath):
                raise FileNotFoundError('The DF_F0.npy file does not exist in this directory.')
        else:
            raise KeyError('This is not a valid data format for ROI alignment.')
        data[s] = np.load(datapath)

    return data


def delete_mac_hidden_files(folder_path):
    folder = Path(folder_path)
    # Patterns to match macOS hidden files
    hidden_files = ['.DS_Store', '._.DS_Store']

    # Recursively find and delete those files
    for hidden_file in hidden_files:
        for file_path in folder.rglob(hidden_file):
            try:
                file_path.unlink()  # Delete the file
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Could not delete {file_path}: {e}")


def load_roicat_results(roicat_dir, roicat_data_name):
    delete_mac_hidden_files(f'{roicat_dir}/{roicat_data_name}.tracking.results_all.richfile')

    results = roicat.util.RichFile_ROICaT(path=f'{roicat_dir}/{roicat_data_name}.tracking.results_all.richfile').load()

    return results


def get_neuron_count(basepath, animal, sessions_to_align):
    # Load iscell data 
    num_valid_neurons = np.zeros(len(sessions_to_align))
    num_neurons = np.zeros(len(sessions_to_align))
    for s, sess in enumerate(sessions_to_align):
        print(f'Loading iscell data for session {sess}')
        datapath = basepath / animal / sess / 'funcimg' / 'Session' / 'suite2p' / 'plane0' / 'iscell.npy'
        if not os.path.exists(datapath):
            raise FileNotFoundError('The iscell.npy file does not exist in this directory.')
        iscell = np.load(datapath)[:,0]
        
        # Select ROIs that are classified as neurons
        neurons = np.where(iscell == 1)[0] 
        num_valid_neurons[s] = len(neurons)
        num_neurons[s] = len(iscell)

    return num_valid_neurons, num_neurons


def visualize_quality_metrics(results):
    # List all available quality metrics
    print('Available quality metrics:')
    print(results['clusters']['quality_metrics'].keys())

    # Plot the distribution of the quality metrics
    confidence = (((np.array(results['clusters']['quality_metrics']['cluster_silhouette']) + 1) / 2) * np.array(results['clusters']['quality_metrics']['cluster_intra_means']))

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15,7))

    axs[0,0].hist(results['clusters']['quality_metrics']['cluster_silhouette'], 50)
    axs[0,0].set_xlabel('cluster_silhouette')
    axs[0,0].set_ylabel('cluster counts')

    axs[0,1].hist(results['clusters']['quality_metrics']['cluster_intra_means'], 50)
    axs[0,1].set_xlabel('cluster_intra_means')
    axs[0,1].set_ylabel('cluster counts')

    axs[1,0].hist(confidence, 50)
    axs[1,0].set_xlabel('confidence')
    axs[1,0].set_ylabel('cluster counts')

    axs[1,1].hist(results['clusters']['quality_metrics']['sample_silhouette'], 50)
    axs[1,1].set_xlabel('sample_silhouette score')
    axs[1,1].set_ylabel('roi sample counts')
    plt.show()

    return fig


def visualize_clusters(results, n_clusters=10):
    # Plot some clusters
    ucids = np.array(results['clusters']['labels'])
    ucids_unique = np.unique(ucids[ucids>=0])

    ROI_ims_sparse = scipy.sparse.vstack(results['ROIs']['ROIs_aligned'])
    ROI_ims_sparse = ROI_ims_sparse.multiply( ROI_ims_sparse.max(1).power(-1) ).tocsr()

    ucid_sfCat = []
    for ucid in ucids_unique:
        idx = np.where(ucids == ucid)[0]
        ucid_sfCat.append( np.concatenate(list(roicat.visualization.crop_cluster_ims(ROI_ims_sparse[idx].toarray().reshape(len(idx), results['ROIs']['frame_height'], results['ROIs']['frame_width']))), axis=1) )

    for ii in range(min(len(ucid_sfCat), n_clusters)):
        fig = plt.figure(figsize=(40,1))
        plt.imshow(ucid_sfCat[ii], cmap='gray')
        plt.axis('off')
    plt.show()

    return fig


def visualize_cluster_persistence(results):
    ucids = np.array(results['clusters']['labels'])
    _, counts = np.unique(ucids, return_counts=True)

    n_sessions = len(results['clusters']['labels_bySession'])
    fig = plt.figure(figsize=(4,4))
    values, bin_edges, _ = plt.hist(counts, bins=n_sessions*2 + 1, range=(0, n_sessions+1))
    plt.xlabel('number of sessions a cluster is present in')
    plt.ylabel('cluster counts')

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    xticks = [bin_centers[np.abs(bin_centers - tick).argmin()] for tick in np.arange(0, n_sessions + 1, 6)]
    plt.xticks(xticks, [str(tick) for tick in np.arange(0, n_sessions + 1, 6)])
    plt.yticks([np.min(values[1:]), np.max(values[1:])])
    plt.show()

    return fig 


def visualize_tracked_clusters_per_session(idx_original_aligned, num_valid_neurons, num_neurons):
    # Get the number of tracked clusters over all (valid) clusters per session 
    num_sessions = idx_original_aligned.shape[0]

    n_tracked_neurons_per_session = np.zeros(num_sessions)
    perc_tracked_over_valid_per_session = np.zeros(num_sessions)
    perc_tracked_over_neurons_per_session = np.zeros(num_sessions)
    for s in range(num_sessions):
        n_tracked_neurons_per_session[s] = len(idx_original_aligned[s][~np.isnan(idx_original_aligned[s])])
        perc_tracked_over_valid_per_session[s] = n_tracked_neurons_per_session[s] / num_valid_neurons[s]
        perc_tracked_over_neurons_per_session[s] = n_tracked_neurons_per_session[s] / num_neurons[s]

    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax = ax.ravel()
    ax[0].plot(np.arange(0, num_sessions), perc_tracked_over_valid_per_session, linewidth=2)
    ax[0].set_yticks([np.round(np.min(perc_tracked_over_valid_per_session), 2), np.round(np.max(perc_tracked_over_valid_per_session), 2)])
    ax[0].set_xticks(np.arange(0, num_sessions + 1, 6))
    ax[0].set_xticklabels(np.arange(0, num_sessions + 1, 6))
    ax[0].set_xlabel('Session')
    ax[0].tick_params(axis='both', labelsize=8)

    ax[0].set_title('% tracked ROIs across all valid ROIs', fontsize=10)

    ax[1].plot(np.arange(0, num_sessions), perc_tracked_over_neurons_per_session, linewidth=2)
    ax[1].set_yticks([np.round(np.min(perc_tracked_over_neurons_per_session), 2), np.round(np.max(perc_tracked_over_neurons_per_session), 2)])
    ax[1].set_xticks(np.arange(0, num_sessions + 1, 6))
    ax[1].set_xticklabels(np.arange(0, num_sessions + 1, 6))
    ax[1].set_xlabel('Session')
    ax[1].tick_params(axis='both', labelsize=8)

    ax[1].set_title('% tracked ROIs across all ROIs', fontsize=10)

    return fig


def visualize_tracked_clusters_in_session(idx_original_aligned):
    # Get the number of tracked clusters per session over all tracked clusters 
    num_sessions = idx_original_aligned.shape[0]
    n_tracked_neurons = idx_original_aligned.shape[1]

    n_tracked_neurons_per_session = np.zeros(num_sessions)
    perc_tracked_per_session_over_tracked = np.zeros(num_sessions)
    for s in range(num_sessions):
        n_tracked_neurons_per_session[s] = len(idx_original_aligned[s][~np.isnan(idx_original_aligned[s])])
        perc_tracked_per_session_over_tracked[s] = n_tracked_neurons_per_session[s] / n_tracked_neurons

    fig = plt.figure(figsize=(4,4))
    plt.plot(np.arange(0, num_sessions), perc_tracked_per_session_over_tracked, linewidth=2)
    plt.xlabel('Session')
    plt.xticks(np.arange(0, num_sessions + 1, 6), labels=np.arange(0, num_sessions + 1, 6), fontsize=8)
    plt.yticks([np.round(np.min(perc_tracked_per_session_over_tracked), 2), np.round(np.max(perc_tracked_per_session_over_tracked), 2)], fontsize=8)
    plt.title('% tracked ROIs per session across all tracked ROIs', fontsize=10)
    plt.show()

    return fig


def visualize_pairwise_cluster_persistence(idx_original_aligned, num_valid_neurons):
    # Get the number of tracked clusters across two sessions over all valid clusters in the first of the two sessions
    num_sessions = idx_original_aligned.shape[0]
    
    idx_array = np.vstack(idx_original_aligned)  # shape: (n_sessions, n_neurons)
    pairwise_cluster_persistence = np.zeros(num_sessions-1)
    for s in range(num_sessions - 1):  # roll over pairs
            pair = idx_array[s:s+2]  # shape: (2, n_neurons)
            valid_mask = np.all(~np.isnan(pair), axis=0)
            valid_indices = np.where(valid_mask)[0]

            pairwise_cluster_persistence[s] = len(valid_indices) / num_valid_neurons[s]

    fig = plt.figure(figsize=(4,4))
    plt.plot(np.arange(0, num_sessions-1), pairwise_cluster_persistence, linewidth=2)
    plt.xlabel('Session')
    plt.xticks(np.arange(0, num_sessions, 6), fontsize=8)
    plt.yticks([np.round(np.min(pairwise_cluster_persistence), 2), np.round(np.max(pairwise_cluster_persistence), 2)], fontsize=8)
    plt.title('% tracked ROIs across two sessions over valid ROIs', fontsize=10)
    plt.show()

    return fig 


def generate_image_slider_html(images, image_size=2, clim=None, interpolation='nearest'):
    """
    Generate an HTML image slider for use in Jupyter Notebooks (returns the HTML instead of displaying it).
    """

    # Check interpolation method
    interpolation_methods = {
        'nearest': Image.Resampling.NEAREST,
        'box': Image.Resampling.BOX,
        'bilinear': Image.Resampling.BILINEAR,
        'hamming': Image.Resampling.HAMMING,
        'bicubic': Image.Resampling.BICUBIC,
        'lanczos': Image.Resampling.LANCZOS,
    }

    if interpolation not in interpolation_methods:
        raise ValueError("Invalid interpolation method.")

    interpolation_method = interpolation_methods[interpolation]

    # Determine image size
    if image_size is None:
        image_size = images[0].shape[:2]
    elif isinstance(image_size, (int, float)):
        image_size = tuple((np.array(images[0].shape[:2]) * image_size).astype(np.int64))
    elif isinstance(image_size, (tuple, list)):
        image_size = tuple(image_size)
    else:
        raise ValueError("Invalid image size.")

    def normalize_image(image, clim=None):
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

        if clim is None:
            clim = (np.min(image), np.max(image))

        norm_image = (image - clim[0]) / (clim[1] - clim[0])
        norm_image = np.clip(norm_image, 0, 1)
        return (norm_image * 255).astype(np.uint8)

    def resize_image(image, new_size, interpolation):
        pil_image = Image.fromarray(image.astype(np.uint8))
        resized_image = pil_image.resize(new_size, resample=interpolation)
        return np.array(resized_image)

    def numpy_to_base64(numpy_array):
        img = Image.fromarray(numpy_array.astype('uint8'))
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("ascii")

    def process_image(image):
        norm_image = normalize_image(image, clim)
        if image_size is not None:
            norm_image = resize_image(norm_image, image_size, interpolation_method)
        return numpy_to_base64(norm_image)

    base64_images = [process_image(img) for img in images]

    slider_id = hashlib.sha256(str(datetime.datetime.now()).encode()).hexdigest()

    html_code = f"""
    <div>
        <input type="range" id="imageSlider_{slider_id}" min="0" max="{len(base64_images) - 1}" value="0">
        <img id="displayedImage_{slider_id}" src="data:image/png;base64,{base64_images[0]}" style="width: {image_size[1]}px; height: {image_size[0]}px;">
        <span id="imageNumber_{slider_id}">Image 0/{len(base64_images) - 1}</span>
    </div>

    <script>
        (function() {{
            let base64_images = {base64_images};
            let current_image = 0;

            function updateImage() {{
                let slider = document.getElementById("imageSlider_{slider_id}");
                current_image = parseInt(slider.value);
                let displayedImage = document.getElementById("displayedImage_{slider_id}");
                displayedImage.src = "data:image/png;base64," + base64_images[current_image];
                let imageNumber = document.getElementById("imageNumber_{slider_id}");
                imageNumber.innerHTML = "Image " + current_image + "/{len(base64_images) - 1}";
            }}

            document.getElementById("imageSlider_{slider_id}").addEventListener("input", updateImage);
        }})();
    </script>
    """
    return HTML(html_code)


def save_gif(images, dir_save, filename):
    
    roicat.helpers.save_gif(
            array=roicat.helpers.add_text_to_images(
                images=[((f / np.max(f)) * 255).astype(np.uint8) for f in images], 
                text=[[f"{ii}",] for ii in range(len(images))], 
                font_size=3,
                line_width=10,
                position=(30, 90),
            ), 
            path=str(Path(dir_save).resolve() / 'visualization' / (filename + '.gif')),
            frameRate=10,
            loop=0,
        )


def visualize_output(results=None, roicat_dir=None, roicat_data_name=None, dir_save=None, 
                     plot_metrics=True, plot_clusters=True, plot_aligned_tracked_clusters=True):
    """Visualize ROICaT output"""

    if dir_save is None:
            raise ValueError('Please provide a directory to save the GIF.')

    # Load results
    if results is None:
        if not roicat_dir or not roicat_data_name:
            raise ValueError("If `results` is not provided, both `roicat_dir` and `roicat_data_name` must be specified.")
        results = load_roicat_results(roicat_dir, roicat_data_name)
 
    # Plot quality metrics
    if plot_metrics:
        _ = visualize_quality_metrics(results)

    # Plot some clusters
    if plot_clusters:
        _ = visualize_clusters(results, n_clusters=10)

    # Display tracked and aligned clusters across sessions
    if plot_aligned_tracked_clusters:
        FOV_clusters = roicat.visualization.compute_colored_FOV(
            labels=np.array(results['clusters']['labels']),
            spatialFootprints=results['ROIs']['ROIs_aligned'], 
            FOV_height=results['ROIs']['frame_height'], 
            FOV_width=results['ROIs']['frame_width'], 
            alphas_sf=np.array(results['clusters']['quality_metrics']['sample_silhouette']) > 0.0,  ## SET INCLUSION CRITERIA FOR CLUSTERS/LABELS
            alphas_labels=np.array(results['clusters']['quality_metrics']['cluster_silhouette']) > 0.0,  ## SET INCLUSION CRITERIA FOR ROI SAMPLES
        )
        # Display slider only if in IPython/Jupyter
        try:
            display(generate_image_slider_html(FOV_clusters, image_size=2))

            save_gif(FOV_clusters, dir_save, filename='tracked_FOV_clusters')

        except:
            print(f'The tracked ROIs cannot be displayed as a slider. Saving a GIF in {dir_save} instead.')
            
            save_gif(FOV_clusters, dir_save, filename='tracked_FOV_clusters')
        

def align_rois(roicat_dir, roicat_data_name, sessions_to_align=None, basepath=None, animal=None, alignment_method='F', 
               neural_data=None, roicat_results=None, plot_alignment=False, save_results=False, force_reload=False,
               alignment_dir=None, filename=None):
    '''Align the neural data according to the ROI they belong to, 
    For more details look at the ROICaT documentation https://roicat.readthedocs.io/en/latest/index.html.'''
    
    # Look for already aligned labels
    if os.path.exists(os.path.join(alignment_dir, filename)) and not force_reload:
        print('Aligned clusters already found. Loading...')
        idx_original_aligned = np.load(os.path.join(alignment_dir, filename))
        return None, idx_original_aligned

    else:
        protocol_nums = [int(re.search(r'protocol-t(\d+)', s).group(1)) for s in sessions_to_align]
        protocol_nums = [protocol_nums[i] - protocol_nums[0] for i in range(len(protocol_nums))]

        # Load neural data
        if neural_data is None:
            neural_data = load_neural_data(basepath, animal, sessions_to_align, data_type=alignment_method)
        
        # Load roicat results
        if roicat_results is None:
            if not roicat_dir or not roicat_data_name:
                raise ValueError("If `results` is not provided, both `roicat_dir` and `roicat_data_name` must be specified.")
            roicat_results = load_roicat_results(roicat_dir, roicat_data_name)

        # Define UCIDs 
        labels_bySession = roicat_results['clusters']['labels_bySession']#.load()
        roi_labels = [rois for rois in labels_bySession[protocol_nums[0]:protocol_nums[-1]+1]]  

        # Update UCIDs with valid cells
        iscell = [[] for s in range(len(sessions_to_align))]
        for s, sess in enumerate(sessions_to_align):
            datapath = basepath / animal / sess / 'funcimg' / 'Session' / 'suite2p' / 'plane0' / 'iscell.npy'
            iscell[s] = np.load(datapath)[:,0]
        
        # Apply the mask to the aligned data
        labels_iscell = roicat.util.mask_UCIDs_with_iscell(ucids=roi_labels, iscell=iscell)
        
        # Squeeze the labels to remove the unassigned labels (not necessary, but reduces the number of unique labels)
        labels_iscell = roicat.util.squeeze_UCID_labels(ucids=labels_iscell, return_array=True)  ## [(n_rois,)] * n_sessions

        # Align the data with the masked labels
        data_aligned_masked, idx_original_aligned = roicat.util.match_arrays_with_ucids(
            arrays=neural_data,  ## expects list (length n_sessions) of numpy arrays (shape (n_rois, n_timepoints))
            ucids=labels_iscell,  ## expects list (length n_sessions) of numpy arrays (shape (n_rois,))  OR   concatenated numpy array (shape (n_rois_total,))
            squeeze=True, return_indices=True
        )

        # Visualize the alignment
        if plot_alignment:
            fig, axs = plt.subplots(2, 2, figsize=(15, 5))
            for i in range(2):
                axs[0, i].imshow(neural_data[i], aspect="auto", cmap="rainbow", interpolation="none")
                axs[1, i].imshow(data_aligned_masked[i], aspect="auto", cmap="rainbow", interpolation="none")
                axs[0, i].set_title(f"Session {i+1}")
                axs[1, i].set_title(f"Session {i+1} (aligned)")
                axs[0, i].set_xlabel("Timepoints")
                axs[1, i].set_xlabel("Timepoints")
                axs[0, i].set_ylabel("ROIs")
                axs[1, i].set_ylabel("ROIs")
                ## Colorbar
                if i == 2 - 1:
                    fig.colorbar(axs[0, i].imshow(neural_data[i], aspect="auto", cmap="rainbow", interpolation="none"), ax=axs[0, i], label="ROI label")
                    fig.colorbar(axs[1, i].imshow(data_aligned_masked[i], aspect="auto", cmap="rainbow", interpolation="none"), ax=axs[1, i], label="ROI label")
            plt.tight_layout()

        # Save indices of aligned data 
        if save_results:
            if alignment_dir is None:
                alignment_dir = os.path.join(basepath, animal)
            else:
                if not os.path.exists(alignment_dir):
                    os.makedirs(alignment_dir)
            if filename is None:
                filename = f"roicat_aligned_ROIs_{'_'.join(['t' + str(n) for n in protocol_nums])}.npy"
            np.save(os.path.join(alignment_dir, filename), idx_original_aligned)

    return data_aligned_masked, idx_original_aligned


def roicat_visualize_tracked_rois(roicat_dir, roicat_data_name, sessions_to_align, roicat_results=None, tracked_neuron_ids=None, dir_save=None, filename=None):
    '''Visualize the alignment of selected ROIs. For more details look at the ROICaT documentation https://roicat.readthedocs.io/en/latest/index.html.'''

    # Handle single or multiple sessions
    if isinstance(sessions_to_align, str):
        sessions_to_align = [sessions_to_align]

    # Get ROICaT results
    if roicat_results is None:
        roicat_results = load_roicat_results(roicat_dir, roicat_data_name)

    # Define UCIDs / labels
    protocol_nums = [int(re.search(r'protocol-t(\d+)', s).group(1)) for s in sessions_to_align]
    protocol_nums = [protocol_nums[i] - protocol_nums[0] for i in range(len(protocol_nums))]
    
    labels_bySession = roicat_results['clusters']['labels_bySession']
    rois_bySession = roicat_results['ROIs']['ROIs_aligned']
    roi_labels = [labels_bySession[p] for p in protocol_nums]
    rois = [rois_bySession[p] for p in protocol_nums]

    if tracked_neuron_ids is not None: # visualize selected neurons
        roi_labels = [np.array(labels)[valid_ids] for labels, valid_ids in zip(roi_labels, tracked_neuron_ids)]
        rois = [labels[valid_ids] for labels, valid_ids in zip(rois, tracked_neuron_ids)]

    # Visualization
    FOV_clusters = roicat.visualization.compute_colored_FOV(
        labels=roi_labels,
        spatialFootprints=rois, 
        FOV_height=roicat_results['ROIs']['frame_height'], 
        FOV_width=roicat_results['ROIs']['frame_width']
    )

    # Display slider only if in IPython/Jupyter
    try:
        display(generate_image_slider_html(FOV_clusters, image_size=2))

        save_gif(FOV_clusters, dir_save, filename=filename)

    except:
        print(f'The tracked ROIs cannot be displayed as a slider. Saving a GIF in {dir_save} instead.')
        
        save_gif(FOV_clusters, dir_save, filename=filename)
    

