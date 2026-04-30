import numpy as np
from scipy.interpolate import interp1d
import copy

class DensityInterpolator1D:
    def __init__(self, density, R):
        """
        Initialize the 1D interpolator with input density and R arrays.

        Parameters
        ----------
        density : array-like
            The array of density values corresponding to R.
        R : array-like
            The array of R values.
        """
        self.original_density = np.array(density)
        self.original_R = np.array(R)
        self._archived_density = copy.deepcopy(self.original_density)
        self._archived_R = copy.deepcopy(self.original_R)
        
        self._corrected_pairs = {}  # key: R_value, value: density
        self._regenerate_interpolator()

    def _regenerate_interpolator(self):
        """Regenerate the interpolator from the current density and R, including corrections."""
        # Start with original arrays
        R = np.copy(self._archived_R)
        density = np.copy(self._archived_density)
        
        # Apply corrections by replacing/adding R-density pairs
        if self._corrected_pairs:
            # Convert corrections to arrays
            corrected_R = np.array(list(self._corrected_pairs.keys()))
            corrected_density = np.array([self._corrected_pairs[r] for r in corrected_R])
            
            # Combine original data with corrections (overwriting conflicts with corrections)
            all_R = np.concatenate([R, corrected_R])
            all_density = np.concatenate([density, corrected_density])
            
            # Get sorted unique R, with preference to corrected densities if duplicated R
            unique_R, indices = np.unique(all_R, return_index=True)
            unique_density = all_density[indices]
        else:
            unique_R = R
            unique_density = density

        # Ensure sorted by R
        sorted_indices = np.argsort(unique_R)
        sorted_R = unique_R[sorted_indices]
        sorted_density = unique_density[sorted_indices]

        self._current_R = sorted_R
        self._current_density = sorted_density
        self.interpolator = interp1d(sorted_R, sorted_density, bounds_error=False, fill_value="extrapolate")

    def interpolate(self, R_values):
        """
        Interpolate density values for given R_values.
        """
        return self.interpolator(R_values)

    def correct(self, R_value, density_value):
        """
        Specify a correction at a specific R value, then regenerate the interpolator.

        Parameters
        ----------
        R_value : float
            The R value for which to override the density.
        density_value : float
            The density value to use at R_value.
        """
        self._corrected_pairs[R_value] = density_value
        self._regenerate_interpolator()

    @property
    def archived_density(self):
        """Get the originally supplied density array."""
        return np.copy(self._archived_density)
    
    @property
    def archived_R(self):
        """Get the originally supplied R array."""
        return np.copy(self._archived_R)

    @property
    def current_density(self):
        """Get the current working density array (including corrections)."""
        return np.copy(self._current_density)
    
    @property
    def current_R(self):
        """Get the current working R array (including corrections)."""
        return np.copy(self._current_R)