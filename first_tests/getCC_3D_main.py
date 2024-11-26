import numpy as np
import matplotlib.pyplot as plt
import os
import json
import h5py
import napari
import pyclesperanto_prototype as cle

from dynamic_cell_tree.connected_components import connected_components_3D
from dynamic_cell_tree.center_finding import find_label_centers_3D
from dynamic_cell_tree.separation import find_label_separation_3D
from dynamic_cell_tree.build_tree import build_trees_from_splits
from dynamic_cell_tree.label_operations import relabel_sequentially_3D
from dynamic_cell_tree.merge_near_centers import merge_close_labels_3D

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

    flow=load_compressed_array(flow_file)
    mask=load_compressed_array(mask_file)

    MetaData_nuclei=get_JSON(nuclei_dir_path)["nuclei_image_MetaData"]
    nuclei_file=os.path.join(nuclei_dir_path,MetaData_nuclei['nuclei image file name'])
    nuclei=getImage(nuclei_file)

    factor = tuple(m / n for m, n in zip(mask.shape, nuclei.shape))
    nuclei=cle.resample(nuclei, factor_x=factor[2], factor_y=factor[1], factor_z=factor[0])

    return mask,flow,nuclei
    return mask[:400,200:800,200:800],flow[:,:400,200:800,200:800],nuclei[:400,200:800,200:800]

def show_results_napari(mask, vector_field, nuclei, labels=None, centers=None):
    """
    Displays the binary mask, vector field, connected components, and label centers in 3D using napari.
    
    Parameters:
        mask (np.ndarray): The binary mask to be displayed.
        vector_field (np.ndarray): The vector field to be displayed.
        nuclei (np.ndarray): Nuclei image to be displayed.
        labels (np.ndarray, optional): Labeled connected components to display.
        centers (dict, optional): Dictionary of label centers (label as key, (x, y, z) coordinates as value).
    """
    # Prepare the grid for starting points in 3D
    X, Y, Z = np.meshgrid(
        np.arange(mask.shape[0]), 
        np.arange(mask.shape[1]), 
        np.arange(mask.shape[2]), 
        indexing='ij'
    )
    start_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T  # (N, 3) array of starting points

    # Unpack vector components for 3D
    U = vector_field[0].ravel()  # X-component
    V = vector_field[1].ravel()  # Y-component
    W = vector_field[2].ravel()  # Z-component

    # Compute the end points by adding the vector directions to start points
    vector_data = np.stack([start_points, np.vstack([U, V, W]).T], axis=1)  # Shape (N, 2, 3)

    # Open napari viewer
    viewer = napari.Viewer()

    # Add the binary mask as an image layer
    viewer.add_image(mask, name="Binary Mask", colormap="gray", opacity=0.5)

    # Add the vector field as a vectors layer
    #viewer.add_vectors(vector_data, edge_color="blue", name="Vector Field")

    # Add connected components as a labels layer
    if labels is not None:
        viewer.add_labels(labels, name="Connected Components")

    # Add nuclei layer
    viewer.add_image(nuclei, name="Nuclei")

    # Add label centers as a points layer if centers are provided
    if centers is not None:
        points = np.array(list(centers.values()))  # Extract the center coordinates
        viewer.add_points(points, name="Label Centers", size=1, face_color="red", symbol="o")

    # Start napari event loop
    napari.run()

import time

def main():
    # Generate example data
    start_time = time.time()
    mask, vector_field, nuclei = generate_example_data_3D()
    print(f"Data generation took {time.time() - start_time:.2f} seconds.")
    
    print("Mask shape:", mask.shape)
    print("Vector field shape:", vector_field.shape)
    print("Nuclei shape:", nuclei.shape)

    # Identify connected components
    start_time = time.time()
    labels = connected_components_3D(mask, vector_field, connectivity=18)
    print(f"Connected components labeling took {time.time() - start_time:.2f} seconds.")
    
    print("Labels shape:", labels.shape)
    print("Max label:", np.max(labels))
    print("Unique labels in connected components:", np.unique(labels))

    # Relabel sequentially
    start_time = time.time()
    labels = relabel_sequentially_3D(labels)
    print(f"Relabeling sequentially took {time.time() - start_time:.2f} seconds.")
    
    print("Unique labels after relabeling:", np.unique(labels))

    # Find label centers
    start_time = time.time()
    centers = find_label_centers_3D(labels, vector_field, connectivity=18)
    print(f"Finding label centers took {time.time() - start_time:.2f} seconds.")
    print("Centers found.")

    # Merge close labels
    start_time = time.time()
    labels = merge_close_labels_3D(labels, centers, merge_distance=4)
    print(f"Merging close labels took {time.time() - start_time:.2f} seconds.")
    
    # Display results with napari
    #show_results_napari(mask, vector_field, nuclei, labels,centers)

    # Find label separation
    start_time = time.time()
    splits = find_label_separation_3D(labels, vector_field, cutoff=20, connectivity=18) #change back if implemented
    print(f"Finding label separation took {time.time() - start_time:.2f} seconds.")
    
    #print("Splits:", splits)

    # Build trees from splits
    start_time = time.time()
    tree = build_trees_from_splits(splits)
    print(f"Building trees from splits took {time.time() - start_time:.2f} seconds.")
    
    #print("Tree structure:", tree)

    # Display final results
    show_results_napari(mask, vector_field, nuclei, labels)



if __name__ == "__main__":
    # Nx=20
    # Ny=10
    # Nz=30
    # mask = np.random.rand(Nx, Ny, Nz) > 0.5
    # vector_field = np.random.rand(3, Nx, Ny, Nz) - 0.5  # Random 3D vectors
    # vector_field[2,:,:,:]=1
    # nuclei = np.random.rand(Nx, Ny, Nz)
    # labels = np.random.randint(0, 4, size=(Nx, Ny, Nz))
    # show_results_napari(mask, vector_field, nuclei, labels)
    main()

