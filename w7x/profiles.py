import numpy as np
from scipy.interpolate import interp1d

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

        self._corrected_pairs = {}  # key: R_value, value: profile
        self._regenerate_interpolator()

    def _regenerate_interpolator(self):
        """Regenerate the interpolator from the original profile and R, including corrections."""
        R = np.copy(self.original_R)
        profile = np.copy(self.original_profile)

        # Apply corrections by replacing/adding R-profile pairs
        if self._corrected_pairs:
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
        self.R = unique_R[sorted_indices]
        self.profile = unique_profile[sorted_indices]

        self.interpolator = interp1d(self.R, self.profile, bounds_error=False, fill_value="extrapolate")

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
