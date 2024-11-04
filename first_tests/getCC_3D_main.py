import numpy as np
import matplotlib.pyplot as plt
import os
import json
import h5py

from dynamic_cell_tree.connected_components import connected_components_3D
from dynamic_cell_tree.center_finding import find_label_centers_3D
from dynamic_cell_tree.separation import find_label_separation
from dynamic_cell_tree.build_tree import build_trees_from_splits
from dynamic_cell_tree.label_operations import relabel_sequentially_3D
from dynamic_cell_tree.merge_near_centers import merge_close_labels

def load_compressed_array(filename):
    # Load the mask, non-zero elements, and original shape from the file
    with h5py.File(filename, 'r') as f:
        mask = f['mask'][:]
        non_zero_elements = f['non_zero_elements'][:]
        shape = f.attrs['shape']

    # Expand the mask to the original shape
    expanded_mask = np.broadcast_to(mask, shape)
    
    # Create an empty array of the original shape
    array = np.zeros(shape, dtype=non_zero_elements.dtype)

    # Reconstruct the original array using the expanded mask and the non-zero elements
    array[expanded_mask] = non_zero_elements
    
    return array

def get_JSON(dir,name=None):
    if name is None:
        name='MetaData.json'
    else:
        name = 'MetaData_'+name+'.json'
    json_file_path=os.path.join(dir, name)
    try:
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        print("MetaData doesn't exist", dir, name)
        data = {}  # Create an empty dictionary if the file doesn't exist
    return data

import tifffile
def getImage(file):
    with tifffile.TiffFile(file) as tif:
            try:
                image=tif.asarray()
            except:
                return None
            
            return image

def generate_example_data_3D():

    apply_folder='20220610_mAG-zGem_H2a-mcherry_78hpf_LM_A3_analyzed_nuclei'
    print(apply_folder)
    applyresult_folder_path=(r'/media/max_kotz/random_data/applyresult/')
    nuclei_folder_path=(r'/media/max_kotz/structured_data/images/RAW_images_and_splitted/raw_images_nuclei/')
    #propresult_folder_path=(r'/media/max_kotz/random_data/propresult/{}').format("")
    #prop_dir_path=os.path.join(propresult_folder_path,apply_folder)
    apply_dir_path=os.path.join(applyresult_folder_path,apply_folder)

    nuclei_dir_path=os.path.join(nuclei_folder_path,apply_folder)
    
    MetaData_apply=get_JSON(apply_dir_path)["apply_MetaData"]
    flow_file=os.path.join(apply_dir_path,MetaData_apply['pred_flows file'])
    mask_file=os.path.join(apply_dir_path,MetaData_apply['segmentation file'])

    flow=load_compressed_array(flow_file)[:,300:400,200:800,200:800]
    mask=load_compressed_array(mask_file)[300:400,200:800,200:800]

    MetaData_nuclei=get_JSON(nuclei_dir_path)["nuclei_image_MetaData"]
    nuclei_file=os.path.join(nuclei_dir_path,MetaData_nuclei['nuclei image file name'])
    nuclei=getImage(nuclei_file)[300:400,200:800,200:800]
    todo: nuclei scalen
    return mask,flow,nuclei

import napari
def show_results_napari(mask, vector_field,nuclei,labels=None):
    """
    Displays the binary mask, vector field, and connected components using napari.
    """

    # Prepare vector data for napari Vectors layer
    X, Y, Z = np.meshgrid(np.arange(mask.shape[0]), np.arange(mask.shape[1]),np.arange(mask.shape[2]))
    start_points = np.vstack([Z.ravel(),Y.ravel(), X.ravel()]).T  # (N^2, 2) array of starting points
    U = vector_field[2, :, :, :].ravel()  # X-component
    V = vector_field[1, :, :,:].ravel()  # Y-component (negative for display alignment)
    W = vector_field[0, :, :,:].ravel()  # Y-component (negative for display alignment)
    
    # Compute end points by adding the direction to the start point
    direction = np.vstack([V, U, W]).T
    vector_data = np.stack([start_points, direction], axis=1)  # Shape (N^2, 2, 2)

    # Open napari viewer
    viewer = napari.Viewer()

    # Add the binary mask as an image layer
    viewer.add_image(mask, name="Binary Mask", colormap="gray", opacity=0.5)

    # Add the vector field as a vectors layer
    viewer.add_vectors(vector_data, edge_color="blue", name="Vector Field")

    # Add connected components as a labels layer
    if not labels is None:
        viewer.add_labels(labels, name="Connected Components")

    viewer.add_image(nuclei,name="nuclei")

    # Start napari event loop
    napari.run()


def main():

    mask, vector_field,nuclei = generate_example_data_3D()
    print(mask.shape)
    print(vector_field.shape)
    print(nuclei.shape)
    show_results_napari(mask,vector_field,nuclei)
    return
    # Identify connected components using the vector field and CCL
    labels = connected_components_3D(mask, vector_field,connectivity=18)
    print('cc3d',np.unique(labels))
    labels=relabel_sequentially_3D(labels)
    print('relaybel',np.unique(labels))

    centers=find_label_centers_3D(labels, vector_field,connectivity=18)
    print(centers)

    labels=merge_close_labels(labels, centers, merge_distance=2)

    splits=find_label_separation(labels, vector_field, cutoff=20,connectivity=18)
    print(splits)
    print(build_trees_from_splits(splits))
    
    show_results_napari(mask,vector_field,nuclei,labels)
    # # Show results with components optionally displayed
    # show_results(mask, vector_field, labels, plot_labels=True)



if __name__ == "__main__":
    main()

