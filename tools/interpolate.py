import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class ProfileInterpolator1D:
    def __init__(self, profile, position):
        """
        Initialize the 1D interpolator with input profile and position arrays.

        Parameters
        ----------
        profile : array-like
            The array of profile values corresponding to ``position``.
        position : array-like
            The array of position values (e.g. R coordinates or position
            indices).
        """
        self.original_profile = np.array(profile)
        self.original_position = np.array(position)

        self.position = np.copy(self.original_position)
        self.profile = np.copy(self.original_profile)
        self.interpolator = interp1d(self.position, self.profile, bounds_error=False, fill_value="extrapolate")

    def reset(self):
        """Reset ``self.position`` / ``self.profile`` to the originals and rebuild the interpolator.

        Discards any corrections that have been applied since construction.
        """
        sorted_indices = np.argsort(self.original_position)
        self.position = np.copy(self.original_position[sorted_indices])
        self.profile = np.copy(self.original_profile[sorted_indices])
        self.interpolator = interp1d(self.position, self.profile, bounds_error=False, fill_value="extrapolate")

    def _regenerate_interpolator(self, idx, profile_value):
        """Overwrite the profile at ``idx`` and rebuild the interpolator."""
        self.profile[idx] = profile_value
        self.interpolator = interp1d(self.position, self.profile, bounds_error=False, fill_value="extrapolate")

    def interpolate(self, position_values):
        """
        Interpolate profile values for given ``position_values``.
        """
        return self.interpolator(position_values)

    def correct(self, position_value, profile_value, by_index=False):
        """
        Override the profile at one or more existing datapoints.

        By default ``position_value`` is interpreted as a position
        coordinate and the correction is applied to the closest
        datapoint in ``self.position``. With ``by_index=True``,
        ``position_value`` is interpreted as a direct index (or
        indices) into ``self.position`` / ``self.profile``.

        Both ``position_value`` and ``profile_value`` may be scalars or
        array-likes of matching length, allowing multiple datapoints to
        be overwritten in a single call.

        Corrections are cumulative: each call modifies the current
        ``self.position`` / ``self.profile`` state, building on any
        previous corrections rather than reapplying them from the
        originals.

        Parameters
        ----------
        position_value : float, int, or array-like
            Position coordinate(s) used to locate the datapoint(s) to
            overwrite, or index/indices into the working arrays when
            ``by_index`` is ``True``.
        profile_value : float or array-like
            New profile value(s) to assign at the selected datapoint(s).
            Must be a scalar or have the same shape as ``position_value``.
        by_index : bool, default False
            If ``True``, treat ``position_value`` as an index (or array
            of indices) rather than a position coordinate.
        """
        position_values = np.atleast_1d(position_value)
        profile_values = np.atleast_1d(profile_value)

        if profile_values.size == 1 and position_values.size > 1:
            profile_values = np.broadcast_to(profile_values, position_values.shape)
        if position_values.shape != profile_values.shape:
            raise ValueError(
                "position_value and profile_value must have matching shapes "
                f"(got {position_values.shape} and {profile_values.shape})."
            )

        if by_index:
            indices = position_values.astype(int)
        else:
            indices = np.array(
                [int(np.argmin(np.abs(self.position - p))) for p in position_values]
            )

        self._regenerate_interpolator(indices, profile_values)

    def add(self, position_value, profile_value):
        """
        Append new datapoints to ``self.position`` / ``self.profile``.

        Both ``position_value`` and ``profile_value`` may be scalars
        (float or int) or array-likes of matching length. The combined
        arrays are re-sorted by position and the interpolator is rebuilt.

        Note that this expands only the working arrays; the originals
        (``self.original_position`` / ``self.original_profile``) are left
        untouched, so :meth:`reset` will discard any added datapoints
        along with corrections.

        Parameters
        ----------
        position_value : float, int, or array-like
            New position coordinate(s) to add.
        profile_value : float or array-like
            Corresponding profile value(s). Must be a scalar or have
            the same shape as ``position_value``.
        """
        position_values = np.atleast_1d(position_value)
        profile_values = np.atleast_1d(profile_value)

        if profile_values.size == 1 and position_values.size > 1:
            profile_values = np.broadcast_to(profile_values, position_values.shape)
        if position_values.shape != profile_values.shape:
            raise ValueError(
                "position_value and profile_value must have matching shapes "
                f"(got {position_values.shape} and {profile_values.shape})."
            )

        new_position = np.concatenate([self.position, position_values])
        new_profile = np.concatenate([self.profile, profile_values])

        sorted_indices = np.argsort(new_position)
        self.position = new_position[sorted_indices]
        self.profile = new_profile[sorted_indices]

        self.interpolator = interp1d(self.position, self.profile, bounds_error=False, fill_value="extrapolate")

    def show(self, show_original=True, ax=None):
        """
        Plot the current position / profile pairs as a solid line.

        Optionally overlays the originally supplied datapoints as a
        scatter plot.

        Parameters
        ----------
        show_original : bool, default True
            If ``True``, scatter the original (``original_position``,
            ``original_profile``) datapoints on top of the line plot.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. If ``None``, a new figure and axes are
            created and ``plt.show()`` is called before returning.

        Returns
        -------
        matplotlib.axes.Axes
            The axes that were drawn on.
        """

        created_fig = ax is None
        if created_fig:
            _, ax = plt.subplots()

        ax.plot(self.position, self.profile, '-', label='profile')
        if show_original:
            ax.scatter(
                self.original_position,
                self.original_profile,
                color='k',
                marker='o',
                zorder=5,
                label='original',
            )

        ax.set_xlabel('Position [A.U]')
        ax.set_ylabel('Profile [A.U]')
        ax.legend()

        if created_fig:
            plt.show()

        return ax
