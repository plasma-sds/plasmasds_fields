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
        self.reset()

    def reset(self):
        """Reset ``self.R`` / ``self.profile`` to the originals and rebuild the interpolator.

        Discards any corrections that have been applied since construction.
        """
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

    def add(self, R_value, profile_value):
        """
        Append new datapoints to ``self.R`` / ``self.profile``.

        Both ``R_value`` and ``profile_value`` may be scalars (float or
        int) or array-likes of matching length. The combined arrays are
        re-sorted by R and the interpolator is rebuilt.

        Note that this expands only the working arrays; the originals
        (``self.original_R`` / ``self.original_profile``) are left
        untouched, so :meth:`reset` will discard any added datapoints
        along with corrections.

        Parameters
        ----------
        R_value : float, int, or array-like
            New R coordinate(s) to add.
        profile_value : float or array-like
            Corresponding profile value(s). Must be a scalar or have
            the same shape as ``R_value``.
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

        new_R = np.concatenate([self.R, R_values])
        new_profile = np.concatenate([self.profile, profile_values])

        sorted_indices = np.argsort(new_R)
        self.R = new_R[sorted_indices]
        self.profile = new_profile[sorted_indices]

        self.interpolator = interp1d(self.R, self.profile, bounds_error=False, fill_value="extrapolate")

    def show(self, show_original=True, ax=None):
        """
        Plot the current R / profile pairs as a solid line.

        Optionally overlays the originally supplied datapoints as a
        scatter plot.

        Parameters
        ----------
        show_original : bool, default True
            If ``True``, scatter the original (``original_R``,
            ``original_profile``) datapoints on top of the line plot.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. If ``None``, a new figure and axes are
            created and ``plt.show()`` is called before returning.

        Returns
        -------
        matplotlib.axes.Axes
            The axes that were drawn on.
        """
        import matplotlib.pyplot as plt

        created_fig = ax is None
        if created_fig:
            _, ax = plt.subplots()

        ax.plot(self.R, self.profile, '-', label='profile')
        if show_original:
            ax.scatter(
                self.original_R,
                self.original_profile,
                color='k',
                marker='o',
                zorder=5,
                label='original',
            )

        ax.set_xlabel('R')
        ax.set_ylabel('profile')
        ax.legend()

        if created_fig:
            plt.show()

        return ax
