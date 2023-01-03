
import numpy as np

def filtering(densities: np.ndarray):
    """Filter the densities with a laplacian filter.
        along y axis, i.e. replace the original densities by 
        the second order derivative of the densities along y axis.
    Args:
        densities (np.ndarray): densities of the scene

    Returns:
        np.ndarray: filtered densities
    """
    results = np.zeros_like(densities)
    
    # first apply a threshold to the densities
    # densities = np.where(densities > 4.6, densities, 0)
    for y in range(1, densities.shape[1] - 1):
        results[:, y, :] = (densities[:, y - 1, :] + densities[:, y + 1, :] - 2 * densities[:, y, :])
    results = (results - results.min())
    return results


