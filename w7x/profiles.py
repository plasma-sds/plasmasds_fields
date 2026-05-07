import numpy as np
from scipy.interpolate import interp1d
import copy

class ProfileInterpolator1D:
    def __init__(self, profile, R):
        """
        Initialize the 1D interpolator with input profile and R arrays.

        Parameters
        ----------
        profile : array-like
            The array of profile values corresponding to R.
        R : array-like
            The array of R values.
        """
        self.original_profile = np.array(profile)
        self.original_R = np.array(R)
        self._archived_profile = copy.deepcopy(self.original_profile)
        self._archived_R = copy.deepcopy(self.original_R)
        
        self._corrected_pairs = {}  # key: R_value, value: profile
        self._regenerate_interpolator()

    def _regenerate_interpolator(self):
        """Regenerate the interpolator from the current profile and R, including corrections."""
        # Start with original arrays
        R = np.copy(self._archived_R)
        profile = np.copy(self._archived_profile)
        
        # Apply corrections by replacing/adding R-profile pairs
        if self._corrected_pairs:
            # Convert corrections to arrays
            corrected_R = np.array(list(self._corrected_pairs.keys()))
            corrected_profile = np.array([self._corrected_pairs[r] for r in corrected_R])
            
            # Combine original data with corrections (overwriting conflicts with corrections)
            all_R = np.concatenate([R, corrected_R])
            all_profile = np.concatenate([profile, corrected_profile])
            
            # Get sorted unique R, with preference to corrected profiles if duplicated R
            unique_R, indices = np.unique(all_R, return_index=True)
            unique_profile = all_profile[indices]
        else:
            unique_R = R
            unique_profile = profile

        # Ensure sorted by R
        sorted_indices = np.argsort(unique_R)
        sorted_R = unique_R[sorted_indices]
        sorted_profile = unique_profile[sorted_indices]

        self._current_R = sorted_R
        self._current_profile = sorted_profile
        self.interpolator = interp1d(sorted_R, sorted_profile, bounds_error=False, fill_value="extrapolate")

    def interpolate(self, R_values):
        """
        Interpolate profile values for given R_values.
        """
        return self.interpolator(R_values)

    def correct(self, R_value, profile_value):
        """
        Specify a correction at a specific R value, then regenerate the interpolator.

        Parameters
        ----------
        R_value : float
            The R value for which to override the profile.
        profile_value : float
            The profile value to use at R_value.
        """
        self._corrected_pairs[R_value] = profile_value
        self._regenerate_interpolator()

    @property
    def archived_profile(self):
        """Get the originally supplied profile array."""
        return np.copy(self._archived_profile)
    
    @property
    def archived_R(self):
        """Get the originally supplied R array."""
        return np.copy(self._archived_R)

    @property
    def current_profile(self):
        """Get the current working profile array (including corrections)."""
        return np.copy(self._current_profile)
    
    @property
    def current_R(self):
        """Get the current working R array (including corrections)."""
        return np.copy(self._current_R)
