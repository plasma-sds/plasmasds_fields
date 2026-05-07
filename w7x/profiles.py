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

        # Working arrays start as a sorted copy of the originals; corrections
        # accumulate by overwriting entries in self.profile in place.
        sorted_indices = np.argsort(self.original_R)
        self.R = np.copy(self.original_R[sorted_indices])
        self.profile = np.copy(self.original_profile[sorted_indices])

        self.interpolator = interp1d(self.R, self.profile, bounds_error=False, fill_value="extrapolate")

    def _regenerate_interpolator(self, idx, profile_value):
        """Overwrite the profile at ``idx`` and rebuild the interpolator."""
        self.profile[idx] = profile_value
        self.interpolator = interp1d(self.R, self.profile, bounds_error=False, fill_value="extrapolate")

    def interpolate(self, R_values):
        """
        Interpolate profile values for given R_values.
        """
        return self.interpolator(R_values)

    def correct(self, R_value, profile_value):
        """
        Override the profile at the existing R datapoint closest to ``R_value``.

        Corrections are cumulative: each call modifies the current
        ``self.R`` / ``self.profile`` state, building on any previous
        corrections rather than reapplying them from the originals.

        Parameters
        ----------
        R_value : float
            The R value used to locate the closest existing datapoint.
        profile_value : float
            The new profile value to assign at that datapoint.
        """
        idx = int(np.argmin(np.abs(self.R - R_value)))
        self._regenerate_interpolator(idx, profile_value)
