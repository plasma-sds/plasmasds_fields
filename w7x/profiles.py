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

        # key: index into original_R, value: corrected profile value
        self._corrections = {}
        self._regenerate_interpolator()

    def _regenerate_interpolator(self):
        """Regenerate the interpolator from the original R and profile, applying corrections in place."""
        R = np.copy(self.original_R)
        profile = np.copy(self.original_profile)

        # Apply corrections by overwriting the profile value at the affected indices
        for idx, value in self._corrections.items():
            profile[idx] = value

        # Ensure sorted by R
        sorted_indices = np.argsort(R)
        self.R = R[sorted_indices]
        self.profile = profile[sorted_indices]

        self.interpolator = interp1d(self.R, self.profile, bounds_error=False, fill_value="extrapolate")

    def interpolate(self, R_values):
        """
        Interpolate profile values for given R_values.
        """
        return self.interpolator(R_values)

    def correct(self, R_value, profile_value):
        """
        Override the profile at the existing R datapoint closest to ``R_value``.

        This does not add a new datapoint. The profile value of the original
        R datapoint nearest to ``R_value`` is replaced with ``profile_value``,
        and the interpolator is regenerated.

        Parameters
        ----------
        R_value : float
            The R value used to locate the closest existing datapoint.
        profile_value : float
            The new profile value to assign at that datapoint.
        """
        idx = int(np.argmin(np.abs(self.original_R - R_value)))
        self._corrections[idx] = profile_value
        self._regenerate_interpolator()
