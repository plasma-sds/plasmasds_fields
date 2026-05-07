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

    def correct(self, R_value, profile_value, by_index=False):
        """
        Override the profile at one or more existing datapoints.

        By default ``R_value`` is interpreted as an R coordinate and the
        correction is applied to the closest datapoint in ``self.R``.
        With ``by_index=True``, ``R_value`` is interpreted as a direct
        index (or indices) into ``self.R`` / ``self.profile``.

        Both ``R_value`` and ``profile_value`` may be scalars or
        array-likes of matching length, allowing multiple datapoints to
        be overwritten in a single call.

        Corrections are cumulative: each call modifies the current
        ``self.R`` / ``self.profile`` state, building on any previous
        corrections rather than reapplying them from the originals.

        Parameters
        ----------
        R_value : float, int, or array-like
            R coordinate(s) used to locate the datapoint(s) to overwrite,
            or index/indices into the working arrays when ``by_index`` is
            ``True``.
        profile_value : float or array-like
            New profile value(s) to assign at the selected datapoint(s).
            Must be a scalar or have the same shape as ``R_value``.
        by_index : bool, default False
            If ``True``, treat ``R_value`` as an index (or array of
            indices) rather than an R coordinate.
        """
        R_values = np.atleast_1d(R_value)
        profile_values = np.atleast_1d(profile_value)

        if profile_values.size == 1 and R_values.size > 1:
            profile_values = np.broadcast_to(profile_values, R_values.shape)
        if R_values.shape != profile_values.shape:
            raise ValueError(
                "R_value and profile_value must have matching shapes "
                f"(got {R_values.shape} and {profile_values.shape})."
            )

        if by_index:
            indices = R_values.astype(int)
        else:
            indices = np.array(
                [int(np.argmin(np.abs(self.R - r))) for r in R_values]
            )

        self._regenerate_interpolator(indices, profile_values)
