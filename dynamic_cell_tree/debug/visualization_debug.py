import numpy as np
import matplotlib.pyplot as plt
import napari

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def show_results(mask, vector_field, labels, plot_labels=True):
    """
    Displays the binary mask, vector field, and connected components with background set to black.
    """
    N = mask.shape[0]
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Plot the binary mask
    axs[0].imshow(mask, cmap="gray")
    axs[0].set_title("Binary Mask of Cells")

    # Plot the vector field
    X, Y = np.meshgrid(np.arange(N), np.arange(N))
    U = vector_field[1, :, :]  # Swapped due to coordinate alignment in quiver
    V = -vector_field[0, :, :] # Negative to align with image Y-axis direction
    axs[1].imshow(mask, cmap="gray", alpha=0.5)
    axs[1].quiver(X, Y, U, V, color="blue", scale=1, scale_units="xy")
    axs[1].set_title("Vector Field (Pointing to Center)")

    # Plot the connected components with a custom colormap if requested
    if plot_labels:
        # Create a custom colormap that maps 0 to black and other labels to Set1
        cmap = plt.cm.get_cmap("Set1", np.max(labels) + 1)
        cmap.set_under("black")  # Color for 0 (background)
        
        # Use 'vmin=0.1' to activate the "under" color for 0 values in labels
        im = axs[2].imshow(labels, cmap=cmap, vmin=0.1)
        axs[2].set_title("Connected Components")
    else:
        axs[2].axis("off")

    plt.show()


def show_results_napari(mask, vector_field, labels):
    """
    Displays the binary mask, vector field, and connected components using napari.
    """
    N = mask.shape[0]

    # Prepare vector data for napari Vectors layer
    X, Y = np.meshgrid(np.arange(N), np.arange(N))
    start_points = np.vstack([Y.ravel(), X.ravel()]).T  # (N^2, 2) array of starting points
    U = vector_field[1, :, :].ravel()  # X-component
    V = vector_field[0, :, :].ravel()  # Y-component (negative for display alignment)

    # Compute end points by adding the direction to the start point
    direction = np.vstack([V, U]).T
    vector_data = np.stack([start_points, direction], axis=1)  # Shape (N^2, 2, 2)

    # Open napari viewer
    viewer = napari.Viewer()

    # Add the binary mask as an image layer
    viewer.add_image(mask, name="Binary Mask", colormap="gray", opacity=0.5)

    # Add the vector field as a vectors layer
    viewer.add_vectors(vector_data, edge_color="blue", name="Vector Field")

    # Add connected components as a labels layer
    viewer.add_labels(labels, name="Connected Components")

    # Start napari event loop
    napari.run()
