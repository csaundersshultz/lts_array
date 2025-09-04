from lts_array.classes.lts_data_class import DataBin
from lts_array.classes.lts_classes import OLSEstimator, LTSEstimator

# Don't print FutureWarning for scipy.lstsq
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


def ltsva(
    st,
    lat_list,
    lon_list,
    elev_list, #TODO This needs to be made optional, not sure the best way to do this. I like the ordering here.
    window_length,
    window_overlap,
    alpha=1.0,
    plot_array_coordinates=False,
    remove_elements=None,
    rij=None,
):
    r"""Process infrasound or seismic array data with least trimmed squares (LTS).

    Args:
        st: Obspy stream object. Assumes response has been removed.
        lat_list (list): List of latitude values for each element in ``st``.
        lon_list (list): List of longitude values for each element in ``st``.
        elev_list (list): List of elevation values for each element in ``st``, measured in meters.
        window_length (float): Window length in seconds.
        window_overlap (float): Window overlap in the range (0.0 - 1.0).
        alpha (float): Fraction of data for LTS subsetting [0.5 - 1.0].
            Choose 1.0 for ordinary least squares (default).
        plot_array_coordinates (bool): Plot array coordinates? Defaults to False.
        remove_elements (list): (Optional) Remove element number(s) from ``st``, ``lat_list``, and ``lon_list`` before processing. Here numbering refers to the Python index (e.g. [0] = remove 1st element in stream).
        rij (array or None): A NumPy array with the first row corresponding to cartesian
            "X" - coordinates and the second row corresponding to cartesian "Y" -
            coordinates, in units of km. If this is provided then ``lat_list`` and
            ``lon_list`` are ignored.

    Returns:
        (tuple):
            A tuple of array processing parameters:
            ``lts_vel`` (array): An array of trace velocity estimates.
            ``lts_baz`` (array): An array of back-azimuth estimates.
            ``t`` (array): An array of times centered on the processing windows.
            ``mdccm`` (array): An array of median cross-correlation maxima.
            ``stdict`` (dict): A dictionary of flagged element pairs.
            ``sigma_tau`` (array): An array of sigma_tau values.
            ``conf_int_vel`` (array): An array of 95% confidence intervals for the trace velocity.
            ``conf_int_baz`` (array): An array of 95% confidence intervals for the back-azimuth.

    """

    # Build data object
    # TODO make DataBin accept elevation list and format it properly DONE
    # TODO make DataBin accept elevations in rij NOT DONE
    data = DataBin(window_length, window_overlap, alpha)
    data.build_data_arrays(st, lat_list, lon_list, elev_list, remove_elements, rij)

    # Plot array coordinates as a check
    # TODO plot array coordinates with elevation as color? DONE
    if plot_array_coordinates:
        data.plot_array_coordinates()

    # TODO make OLSE return azimuth and zenith source angles
    # TODO make LTSE return azimuth and zenith source angles
    if data.alpha == 1.0:
        # Ordinary Least Squares
        ltsva = OLSEstimator(data)
    else:
        # Least Trimmed Squares
        ltsva = LTSEstimator(data)
    ltsva.correlate(data)  # Note, correlate results are consistent with/without elev_list, as it should be
    if len(data.rij) == 2:  # 2d case
        ltsva.solve(data)
        return (
            ltsva.lts_vel,
            ltsva.lts_baz,
            ltsva.t,
            ltsva.mdccm,
            ltsva.stdict,
            ltsva.sigma_tau,
            ltsva.conf_int_vel,
            ltsva.conf_int_baz,
        )  # Does not include elevation angle
    elif len(data.rij) == 3:  # 3d case
        ltsva.solve_3d(data)
        return (
            ltsva.lts_vel,
            ltsva.lts_baz,
            ltsva.lts_elev,
            ltsva.t,
            ltsva.mdccm,
            ltsva.stdict,
            ltsva.sigma_tau,
            #ltsva.conf_int_vel,
            #ltsva.conf_int_baz,
            #ltsva.conf_int_elev,
        )  # Confidence intervals not updated yet
