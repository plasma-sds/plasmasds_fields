"""
1-D profile interpolator with editable datapoints.
"""
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class ProfileInterpolator1D:
    """
    Linear 1-D interpolator of a profile, with editable datapoints.

    The profile is interpolated linearly between the datapoints and
    extrapolated linearly outside them. Datapoints can be corrected
    (:meth:`correct_datapoints`) or added (:meth:`add_datapoints`), and
    :meth:`reset` restores the original data. Calling the instance interpolates
    the profile, so it can be passed directly as ``density_function`` to
    :func:`w7x.fields.add_density`.

    Parameters
    ----------
    position : array_like
        Positions of the datapoints, e.g. R coordinates [m]. They do not need
        to be sorted.
    profile : array_like
        Profile values at ``position``, e.g. densities [m^-3].

    Attributes
    ----------
    position, profile : numpy.ndarray
        Current datapoints, including corrections and added points. Indices
        follow the order of the input arrays, with added datapoints appended at
        the end.
    original_position, original_profile : numpy.ndarray
        Datapoints as given at construction.
    interpolator : scipy.interpolate.interp1d
        Interpolator built from the current datapoints.
    """

    def __init__(self,  position, profile):
        self.original_profile = np.array(profile, dtype=float)
        self.original_position = np.array(position, dtype=float)

        self.position = np.copy(self.original_position)
        self.profile = np.copy(self.original_profile)
        self.interpolator = interp1d(self.position, self.profile, bounds_error=False, fill_value="extrapolate")

    def __call__(self, position_values):
        """
        Interpolate the profile at ``position_values``.

        Equivalent to ``self.interpolator(position_values)``.

        Parameters
        ----------
        position_values : float or array_like
            Positions at which to evaluate the profile.

        Returns
        -------
        numpy.ndarray
            Interpolated (or extrapolated) profile values.
        """
        return self.interpolator(position_values)

    def _regenerate_interpolator(self, idx, profile_value):
        """Overwrite the profile at ``idx`` and rebuild the interpolator."""
        self.profile[idx] = profile_value
        self.interpolator = interp1d(self.position, self.profile, bounds_error=False, fill_value="extrapolate")

    def correct_datapoints(self, position_value, profile_value, by_index=False):
        """
        Overwrite the profile at one or more existing datapoints.

        By default ``position_value`` is a position, and the closest datapoint
        in ``self.position`` is corrected. With ``by_index=True``,
        ``position_value`` is an index (or indices) into ``self.position`` /
        ``self.profile``. Corrections are cumulative: each call builds on the
        current datapoints, including earlier corrections.

        Parameters
        ----------
        position_value : float, int or array_like
            Position(s) of the datapoint(s) to overwrite, or their
            index/indices if ``by_index`` is True.
        profile_value : float or array_like
            New profile value(s). Either a scalar or the same shape as
            ``position_value``.
        by_index : bool, default False
            If True, treat ``position_value`` as index/indices rather than
            positions. Indices follow the order of the arrays given at
            construction, with datapoints from :meth:`add_datapoints` appended
            at the end.

        Raises
        ------
        ValueError
            If ``profile_value`` is neither a scalar nor the same shape as
            ``position_value``.
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

    def add_datapoints(self, position_value, profile_value):
        """
        Append new datapoints and rebuild the interpolator.

        New datapoints are appended at the end, so existing indices are
        unchanged. Only the current datapoints are extended: :meth:`reset`
        discards added datapoints along with corrections.

        Parameters
        ----------
        position_value : float, int or array_like
            Position(s) of the new datapoint(s).
        profile_value : float or array_like
            Profile value(s) of the new datapoint(s). Either a scalar or the
            same shape as ``position_value``.

        Raises
        ------
        ValueError
            If ``profile_value`` is neither a scalar nor the same shape as
            ``position_value``.
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

        self.position = np.concatenate([self.position, position_values])
        self.profile = np.concatenate([self.profile, profile_values])

        self.interpolator = interp1d(self.position, self.profile, bounds_error=False, fill_value="extrapolate")

    def reset(self):
        """
        Restore the original datapoints and rebuild the interpolator.

        Discards all corrections and added datapoints.
        """
        self.position = np.copy(self.original_position)
        self.profile = np.copy(self.original_profile)
        self.interpolator = interp1d(self.position, self.profile, bounds_error=False, fill_value="extrapolate")

    def show(self, show_original=True, ax=None):
        """
        Plot the current profile, optionally with the original datapoints.

        Parameters
        ----------
        show_original : bool, default True
            If True, overlay the original datapoints as a scatter plot.
        ax : matplotlib.axes.Axes, default None
            Axes to draw on. If None, a new figure is created and
            ``plt.show()`` is called before returning.

        Returns
        -------
        matplotlib.axes.Axes
            The axes that were drawn on.
        """

        created_fig = ax is None
        if created_fig:
            _, ax = plt.subplots()

        order = np.argsort(self.position)
        ax.plot(self.position[order], self.profile[order], '-', label='profile')
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
