from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple
import warnings

import gsw
import numpy as np
import xarray as xr


CP_SEAWATER = 3992.0
GRAVITY = 9.81
SEA_ICE_SALINITY = 5.0


def _apply_gsw(func, *args, output_count: int = 1):
    kwargs = {
        "dask": "parallelized",
        "output_dtypes": [float] * output_count,
        "keep_attrs": True,
    }
    if output_count > 1:
        kwargs["output_core_dims"] = [[] for _ in range(output_count)]
    return xr.apply_ufunc(func, *args, **kwargs)


def _with_attrs(da, *, name: str, units: str, long_name: str, comment: str):
    out = da.copy()
    out.name = name
    out.attrs.update(
        {
            "units": units,
            "long_name": long_name,
            "comment": comment,
        }
    )
    return out


def _as_dataarray_or_value(value):
    return value


def _maybe_flip_flux(flux, *, positive_down: bool):
    return flux if positive_down else -flux


def _broadcast_lon_lat(lon, lat, template):
    if lon is None or lat is None:
        return lon, lat
    if not isinstance(template, xr.DataArray):
        return lon, lat
    lon_da = lon if isinstance(lon, xr.DataArray) else xr.DataArray(lon)
    lat_da = lat if isinstance(lat, xr.DataArray) else xr.DataArray(lat)
    lon_da, lat_da = xr.broadcast(lon_da, lat_da)
    return lon_da, lat_da


def to_teos10(
    salinity,
    temperature,
    p=0,
    *,
    lon=None,
    lat=None,
    input_kind: str = "SP_pt",
    warn_without_location: bool = True,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Convert common CMIP-style ocean inputs to TEOS-10 variables.

    Parameters
    ----------
    salinity
        By default, practical salinity such as CMIP ``so``.
    temperature
        By default, potential temperature such as CMIP ``thetao``.
    p
        Sea pressure in dbar. Use 0 for surface fields.
    lon, lat
        Longitude and latitude used by ``gsw.SA_from_SP``. If these are omitted
        and ``input_kind='SP_pt'``, the function uses ``SA ~= SP`` as an
        approximation and emits a warning.
    input_kind
        ``'SP_pt'`` for practical salinity + potential temperature, or
        ``'SA_CT'`` when the inputs are already Absolute Salinity and
        Conservative Temperature.

    Returns
    -------
    SA, CT
        Absolute Salinity and Conservative Temperature.
    """
    if input_kind == "SA_CT":
        return salinity, temperature
    if input_kind != "SP_pt":
        raise ValueError("input_kind must be either 'SP_pt' or 'SA_CT'")

    SP = _as_dataarray_or_value(salinity)
    pt = _as_dataarray_or_value(temperature)

    if lon is None or lat is None:
        if warn_without_location:
            warnings.warn(
                "lon/lat were not provided; using SA ~= SP before converting thetao to CT. "
                "Provide lon and lat for a TEOS-10 Absolute Salinity correction.",
                UserWarning,
                stacklevel=2,
            )
        SA = SP
    else:
        lon_b, lat_b = _broadcast_lon_lat(lon, lat, SP)
        SA = _apply_gsw(gsw.SA_from_SP, SP, p, lon_b, lat_b)
        if isinstance(SA, xr.DataArray):
            SA = _with_attrs(
                SA,
                name="SA",
                units="g kg-1",
                long_name="Absolute Salinity",
                comment="Converted from practical salinity using gsw.SA_from_SP.",
            )

    CT = _apply_gsw(gsw.CT_from_pt, SA, pt)
    if isinstance(CT, xr.DataArray):
        CT = _with_attrs(
            CT,
            name="CT",
            units="degC",
            long_name="Conservative Temperature",
            comment="Converted from potential temperature using gsw.CT_from_pt.",
        )
    return SA, CT


def thermodynamic_coefficients(
    salinity,
    temperature,
    p=0,
    *,
    lon=None,
    lat=None,
    input_kind: str = "SP_pt",
):
    """
    Return TEOS-10 density, thermal expansion, and haline contraction.

    By default this accepts CMIP-style ``so`` and ``thetao`` and converts them
    to ``SA`` and ``CT`` first. Use ``input_kind='SA_CT'`` to skip conversion.
    """
    SA, CT = to_teos10(salinity, temperature, p, lon=lon, lat=lat, input_kind=input_kind)
    rho, alpha, beta = _apply_gsw(gsw.rho_alpha_beta, SA, CT, p, output_count=3)
    if isinstance(rho, xr.DataArray):
        rho = _with_attrs(rho, name="rho", units="kg m-3", long_name="In-situ density", comment="Calculated with gsw.rho_alpha_beta.")
        alpha = _with_attrs(alpha, name="alpha", units="K-1", long_name="Thermal expansion coefficient", comment="Calculated with gsw.rho_alpha_beta.")
        beta = _with_attrs(beta, name="beta", units="kg g-1", long_name="Haline contraction coefficient", comment="Calculated with gsw.rho_alpha_beta.")
    return rho, alpha, beta


def density_from_salinity_temperature(
    salinity,
    temperature,
    p=0,
    *,
    lon=None,
    lat=None,
    input_kind: str = "SP_pt",
    density_kind: str = "rho",
):
    """
    Calculate density used for WMT binning from salinity and temperature.

    Parameters
    ----------
    density_kind
        ``'rho'`` returns in-situ density in kg m-3. This matches bins such as
        1024-1029. ``'sigma0'`` returns potential density anomaly referenced to
        0 dbar. This matches bins such as 24-29.
    """
    SA, CT = to_teos10(salinity, temperature, p, lon=lon, lat=lat, input_kind=input_kind)
    normalized = str(density_kind).lower()
    if normalized in {"rho", "density", "in_situ", "in-situ"}:
        density = _apply_gsw(gsw.rho, SA, CT, p)
        if isinstance(density, xr.DataArray):
            density = _with_attrs(
                density,
                name="rho",
                units="kg m-3",
                long_name="In-situ density",
                comment="Calculated from salinity and temperature using gsw.rho.",
            )
        return density
    if normalized in {"sigma0", "sigma_0", "potential_density_anomaly"}:
        density = _apply_gsw(gsw.sigma0, SA, CT)
        if isinstance(density, xr.DataArray):
            density = _with_attrs(
                density,
                name="sigma0",
                units="kg m-3",
                long_name="Potential density anomaly referenced to 0 dbar",
                comment="Calculated from salinity and temperature using gsw.sigma0.",
            )
        return density
    raise ValueError("density_kind must be either 'rho' or 'sigma0'")


def buoyancy_flux(
    H,
    FWF,
    S,
    T,
    p=0,
    *,
    lon=None,
    lat=None,
    input_kind: str = "SP_pt",
    formula: str = "default",
    heat_positive_down: bool = True,
    fwf_positive_down: bool = True,
    cp: float = CP_SEAWATER,
    gravity: float = GRAVITY,
):
    """
    Calculate buoyancy flux from heat and freshwater fluxes.

    This function is compatible with the old ``wmt.py`` call style:
    ``buoyancy_flux(H, FWF, so, thetao, p)``. The default assumes ``S`` is
    practical salinity and ``T`` is potential temperature.

    Sign convention
    ---------------
    ``heat_positive_down=True`` means positive heat flux enters the ocean.
    ``fwf_positive_down=True`` means positive freshwater flux enters the ocean.
    Set either flag to False if your input uses the opposite convention.

    Returns
    -------
    B, B_HF, B_FWF
        Total, heat-driven, and freshwater-driven buoyancy flux terms.
    """
    H = _maybe_flip_flux(H, positive_down=heat_positive_down)
    FWF = _maybe_flip_flux(FWF, positive_down=fwf_positive_down)
    rho, alpha, beta = thermodynamic_coefficients(S, T, p, lon=lon, lat=lat, input_kind=input_kind)

    if formula == "default":
        B_HF = -gravity / rho * (alpha * H / cp)
        B_FWF = -gravity / rho * (beta * FWF * S)
        units = "m2 s-3"
        standard = "surface_buoyancy_flux"
    elif formula == "hf_term":
        B_HF = H
        B_FWF = beta * FWF * S * cp / alpha
        units = "W m-2"
        standard = "equivalent_heat_flux"
    else:
        raise ValueError("formula must be either 'default' or 'hf_term'")

    B = B_HF + B_FWF
    comment = (
        "Calculated from heat and freshwater fluxes. Input salinity/temperature "
        f"interpreted as {input_kind}; heat_positive_down={heat_positive_down}; "
        f"fwf_positive_down={fwf_positive_down}."
    )
    if isinstance(B, xr.DataArray):
        B = _with_attrs(B, name="buoyancy_flux", units=units, long_name="Buoyancy flux", comment=comment)
        B.attrs["standard_name"] = standard
        B_HF = _with_attrs(B_HF, name="buoyancy_flux_heat", units=units, long_name="Heat contribution to buoyancy flux", comment=comment)
        B_FWF = _with_attrs(B_FWF, name="buoyancy_flux_freshwater", units=units, long_name="Freshwater contribution to buoyancy flux", comment=comment)
    return B, B_HF, B_FWF


def buoyancy_flux_heat(
    H,
    S,
    T,
    p=0,
    *,
    lon=None,
    lat=None,
    input_kind: str = "SP_pt",
    heat_positive_down: bool = True,
    cp: float = CP_SEAWATER,
    gravity: float = GRAVITY,
):
    H = _maybe_flip_flux(H, positive_down=heat_positive_down)
    rho, alpha, _ = thermodynamic_coefficients(S, T, p, lon=lon, lat=lat, input_kind=input_kind)
    bf = -gravity / rho * (alpha * H / cp)
    if isinstance(bf, xr.DataArray):
        bf = _with_attrs(bf, name="buoyancy_flux_heat", units="m2 s-3", long_name="Heat contribution to buoyancy flux", comment="Positive heat flux is assumed downward unless heat_positive_down=False.")
    return bf


def buoyancy_flux_water(
    FWF,
    S,
    T,
    p=0,
    *,
    lon=None,
    lat=None,
    input_kind: str = "SP_pt",
    fwf_positive_down: bool = True,
    gravity: float = GRAVITY,
):
    FWF = _maybe_flip_flux(FWF, positive_down=fwf_positive_down)
    rho, _, beta = thermodynamic_coefficients(S, T, p, lon=lon, lat=lat, input_kind=input_kind)
    bf = -gravity / rho * (beta * FWF * S)
    if isinstance(bf, xr.DataArray):
        bf = _with_attrs(bf, name="buoyancy_flux_freshwater", units="m2 s-3", long_name="Freshwater contribution to buoyancy flux", comment="Positive freshwater flux is assumed downward unless fwf_positive_down=False.")
    return bf


def buoyancy_flux_water_to_heat(
    FWF,
    S,
    T,
    p=0,
    *,
    lon=None,
    lat=None,
    input_kind: str = "SP_pt",
    fwf_positive_down: bool = True,
    cp: float = CP_SEAWATER,
):
    FWF = _maybe_flip_flux(FWF, positive_down=fwf_positive_down)
    _, alpha, beta = thermodynamic_coefficients(S, T, p, lon=lon, lat=lat, input_kind=input_kind)
    hf = beta * FWF * S * cp / alpha
    if isinstance(hf, xr.DataArray):
        hf = _with_attrs(hf, name="freshwater_equivalent_heat_flux", units="W m-2", long_name="Freshwater contribution expressed as equivalent heat flux", comment="Positive freshwater flux is assumed downward unless fwf_positive_down=False.")
    return hf


def trans_rate_heat(
    H,
    S,
    T,
    p=0,
    *,
    lon=None,
    lat=None,
    input_kind: str = "SP_pt",
    heat_positive_down: bool = True,
    cp: float = CP_SEAWATER,
):
    H = _maybe_flip_flux(H, positive_down=heat_positive_down)
    _, alpha, _ = thermodynamic_coefficients(S, T, p, lon=lon, lat=lat, input_kind=input_kind)
    tr = -alpha * H / cp
    if isinstance(tr, xr.DataArray):
        tr = _with_attrs(tr, name="transformation_rate_heat", units="kg m-2 s-1", long_name="Heat-driven transformation rate", comment="Uses alpha * heat flux / cp with the configured heat flux sign convention.")
    return tr


def trans_rate_water(
    FWF,
    S,
    T,
    p=0,
    *,
    lon=None,
    lat=None,
    input_kind: str = "SP_pt",
    fwf_positive_down: bool = True,
):
    FWF = _maybe_flip_flux(FWF, positive_down=fwf_positive_down)
    _, _, beta = thermodynamic_coefficients(S, T, p, lon=lon, lat=lat, input_kind=input_kind)
    tr = -beta * FWF * S
    if isinstance(tr, xr.DataArray):
        tr = _with_attrs(tr, name="transformation_rate_freshwater", units="kg m-2 s-1", long_name="Freshwater-driven transformation rate", comment="Uses beta * freshwater flux * salinity with the configured freshwater flux sign convention.")
    return tr


def trans_rate_seaice(
    SeaiceWF,
    S,
    T,
    p=0,
    *,
    lon=None,
    lat=None,
    input_kind: str = "SP_pt",
    seaice_salinity: float = SEA_ICE_SALINITY,
    seaice_positive_down: bool = True,
):
    SeaiceWF = _maybe_flip_flux(SeaiceWF, positive_down=seaice_positive_down)
    _, _, beta = thermodynamic_coefficients(S, T, p, lon=lon, lat=lat, input_kind=input_kind)
    correction = 1 - seaice_salinity / S
    tr = -beta * correction * S * SeaiceWF
    if isinstance(tr, xr.DataArray):
        tr = _with_attrs(tr, name="transformation_rate_seaice", units="kg m-2 s-1", long_name="Sea-ice freshwater transformation rate", comment=f"Sea-ice salinity correction uses seaice_salinity={seaice_salinity}.")
    return tr


def trans_rate_mix_sal(
    U,
    V,
    S,
    T,
    p=0,
    *,
    dx=None,
    dy=None,
    x_dim: str = "x",
    y_dim: str = "y",
    lon=None,
    lat=None,
    input_kind: str = "SP_pt",
    cp: float = CP_SEAWATER,
):
    """
    Estimate salinity-advection/mixing transformation.

    This keeps the old function name but avoids assuming that axis 1 and axis 2
    are the horizontal grid. Pass ``dx`` and ``dy`` in metres whenever possible.
    If they are omitted, xarray coordinate spacing is used, which is only safe
    for regular Cartesian grids.
    """
    if not isinstance(S, xr.DataArray):
        raise TypeError("trans_rate_mix_sal now expects S to be an xarray.DataArray so dimensions are explicit")

    if dx is None or dy is None:
        warnings.warn(
            "dx/dy were not provided; using coordinate spacing from S.differentiate(). "
            "For CMIP curvilinear ocean grids, pass grid metrics dx and dy.",
            UserWarning,
            stacklevel=2,
        )
        dSdx = S.differentiate(x_dim)
        dSdy = S.differentiate(y_dim)
    else:
        dSdx = S.diff(x_dim) / dx
        dSdy = S.diff(y_dim) / dy
        dSdx = dSdx.reindex({x_dim: S[x_dim]}, method="nearest")
        dSdy = dSdy.reindex({y_dim: S[y_dim]}, method="nearest")

    rho, _, beta = thermodynamic_coefficients(S, T, p, lon=lon, lat=lat, input_kind=input_kind)
    tr = -beta * S * (U * dSdx + V * dSdy) * cp * rho
    return _with_attrs(
        tr,
        name="transformation_rate_salinity_advection",
        units="W m-3",
        long_name="Salinity-gradient contribution to transformation",
        comment="Requires careful grid metrics on curvilinear ocean grids.",
    )


LAT_COORD_CANDIDATES = (
    "lat",
    "latitude",
    "nav_lat",
    "TLAT",
    "yt_ocean",
    "geolat",
)
LON_COORD_CANDIDATES = (
    "lon",
    "longitude",
    "nav_lon",
    "TLONG",
    "xt_ocean",
    "geolon",
)


def get_lat_lon_coords(obj, *, required: bool = True):
    """
    Return likely latitude and longitude coordinate names.

    The function checks common CMIP ocean-grid names. It works with either a
    DataArray or Dataset, as long as the coordinates/variables are attached to
    the object.
    """
    names = set(obj.coords)
    if isinstance(obj, xr.Dataset):
        names.update(obj.data_vars)

    lat_name = next((name for name in LAT_COORD_CANDIDATES if name in names), None)
    lon_name = next((name for name in LON_COORD_CANDIDATES if name in names), None)
    if required and (lat_name is None or lon_name is None):
        raise ValueError(
            "Could not infer latitude/longitude coordinates. "
            f"Available coords: {list(obj.coords)}"
        )
    return lat_name, lon_name


def _coord_or_var(obj, name):
    if obj is None or name is None:
        return None
    if name in obj.coords:
        return obj.coords[name]
    if isinstance(obj, xr.Dataset) and name in obj.data_vars:
        return obj[name]
    return None


def _infer_lon_lat(*arrays, lon=None, lat=None):
    if lon is not None and lat is not None:
        return lon, lat

    found_lon = lon
    found_lat = lat
    for arr in arrays:
        if arr is None or not isinstance(arr, (xr.DataArray, xr.Dataset)):
            continue
        lat_name, lon_name = get_lat_lon_coords(arr, required=False)
        if found_lat is None:
            found_lat = _coord_or_var(arr, lat_name)
        if found_lon is None:
            found_lon = _coord_or_var(arr, lon_name)
        if found_lon is not None and found_lat is not None:
            return found_lon, found_lat
    return found_lon, found_lat


def _infer_time_dim(arr, time_dim=None):
    if time_dim is not None:
        if time_dim not in arr.dims:
            raise ValueError(f"Expected time_dim {time_dim!r} in dims={arr.dims}")
        return time_dim
    for candidate in ("time", "month"):
        if candidate in arr.dims:
            return candidate
    raise ValueError(f"Cannot infer time dimension from dims={arr.dims}. Pass time_dim explicitly.")


def _apply_lat_boundary(arr, lat, lat_boundary):
    if lat_boundary is None or arr is None:
        return arr
    if lat is None:
        raise ValueError("lat_boundary was provided, but no latitude coordinate could be inferred")
    return arr.where(lat < lat_boundary)


def _density_centers(density_bins, density_min, density_max, density_step):
    if density_bins is not None:
        centers = np.asarray(density_bins, dtype=float)
        if centers.ndim != 1 or centers.size == 0:
            raise ValueError("density_bins must be a non-empty 1D sequence of bin centers")
        if density_step is None:
            if centers.size < 2:
                raise ValueError("density_step is required when only one density bin center is provided")
            density_step = float(np.median(np.diff(centers)))
        return centers, float(density_step)

    if density_min is None or density_max is None:
        raise ValueError("Provide either density_bins or density_min/density_max")
    if density_step <= 0:
        raise ValueError("density_step must be positive")
    return np.arange(density_min, density_max, density_step), float(density_step)


def _normalize_transform_type(transform_type):
    normalized = str(transform_type).lower().replace("_", " ").replace("-", " ").strip()
    aliases = {
        "freshwater": "water",
        "fwf": "water",
        "water": "water",
        "heat": "heat",
        "sea ice": "seaice",
        "seaice": "seaice",
        "sid": "seaice",
        "sidd": "seaice",
        "raw": "raw",
        "none": "raw",
    }
    if normalized not in aliases:
        raise ValueError("transform_type must be one of: 'water', 'heat', 'seaice', or 'raw'")
    return aliases[normalized]


def _horizontal_dims(arr):
    if not isinstance(arr, xr.DataArray) or arr.ndim < 2:
        return ()
    non_horizontal = {"time", "month", "block", "lev", "depth", "rho", "density"}
    candidates = [dim for dim in arr.dims if dim not in non_horizontal]
    if len(candidates) >= 2:
        return tuple(candidates[-2:])
    return tuple(arr.dims[-2:])


def _maybe_rename_area_dims(area, target):
    if not isinstance(area, xr.DataArray):
        area = xr.DataArray(area)
    target_dims = _horizontal_dims(target)
    if area.ndim != 2 or len(target_dims) != 2:
        return area
    if set(area.dims).issubset(set(target.dims)):
        return area
    target_shape = tuple(target.sizes[dim] for dim in target_dims)
    if tuple(area.shape) != target_shape:
        raise ValueError(
            "2D area shape does not match target horizontal shape. "
            f"area dims={area.dims}, area shape={area.shape}, "
            f"target horizontal dims={target_dims}, target horizontal shape={target_shape}. "
            "Pre-align/regrid area explicitly before calling wmt_by_density_bins()."
        )
    return area.rename({old: new for old, new in zip(area.dims, target_dims)})


def align_area_to_data(area, target, mode="auto"):
    """
    Align a cell-area field to a transformation-rate field.

    This function intentionally does not regrid. It only performs xarray
    alignment, broadcasting, and a conservative 2D dimension rename when the
    horizontal shape exactly matches the target's horizontal dimensions.
    """
    if area is None:
        return None
    area = _maybe_rename_area_dims(area, target)

    if mode == "exact":
        _, aligned_area = xr.align(target, area, join="exact")
        return aligned_area
    if mode == "inner":
        _, aligned_area = xr.align(target, area, join="inner")
        return aligned_area
    if mode != "auto":
        raise ValueError("area_align must be one of: 'auto', 'exact', or 'inner'")

    try:
        _, aligned_area = xr.align(target, area, join="exact")
        return aligned_area
    except ValueError:
        pass

    try:
        _, aligned_area = xr.align(target, area, join="inner")
        return aligned_area
    except ValueError as exc:
        raise ValueError(
            "Could not align area to data. This likely means the area grid and "
            "the variable grid differ. Pre-align/regrid area explicitly before "
            "calling wmt_by_density_bins(). "
            f"target dims={target.dims}, area dims={area.dims}"
        ) from exc


def _apply_area_and_output_unit(wmt_rate, area, output_unit, area_align):
    normalized = "native" if output_unit is None else str(output_unit).lower()
    if normalized in {"native", "none", "raw"}:
        if area is not None:
            wmt_rate = wmt_rate * align_area_to_data(area, wmt_rate, mode=area_align)
            wmt_rate.attrs["area_weighted"] = True
        return wmt_rate, wmt_rate.attrs.get("units", "native")

    if normalized != "sv":
        raise ValueError("output_unit must be one of: None, 'native', or 'Sv'")
    if area is None:
        raise ValueError("area is required when output_unit='Sv'")

    aligned_area = align_area_to_data(area, wmt_rate, mode=area_align)
    out = wmt_rate * aligned_area / 1e6
    out.attrs["units"] = "Sv"
    out.attrs["area_weighted"] = True
    return out, "Sv"


def _normalize_wmt_sign(wmt_sign):
    normalized = str(wmt_sign).lower().replace("_", " ").replace("-", " ").strip()
    aliases = {
        "positive densification": "positive_densification",
        "densification": "positive_densification",
        "dense": "positive_densification",
        "positive lightening": "positive_lightening",
        "lightening": "positive_lightening",
        "freshening": "positive_lightening",
    }
    if normalized not in aliases:
        raise ValueError("wmt_sign must be 'positive_densification' or 'positive_lightening'")
    return aliases[normalized]


def _transformation_from_forcing(
    forcing,
    salinity,
    temperature,
    p,
    *,
    transform_type,
    lon=None,
    lat=None,
    input_kind="SP_pt",
    heat_positive_down=True,
    fwf_positive_down=True,
    seaice_positive_down=True,
):
    transform_type = _normalize_transform_type(transform_type)
    if transform_type == "raw":
        return forcing
    if salinity is None or temperature is None:
        raise ValueError("salinity and temperature are required unless transform_type='raw'")
    if transform_type == "water":
        return trans_rate_water(
            forcing,
            salinity,
            temperature,
            p=p,
            lon=lon,
            lat=lat,
            input_kind=input_kind,
            fwf_positive_down=fwf_positive_down,
        )
    if transform_type == "heat":
        return trans_rate_heat(
            forcing,
            salinity,
            temperature,
            p=p,
            lon=lon,
            lat=lat,
            input_kind=input_kind,
            heat_positive_down=heat_positive_down,
        )
    return trans_rate_seaice(
        forcing,
        salinity,
        temperature,
        p=p,
        lon=lon,
        lat=lat,
        input_kind=input_kind,
        seaice_positive_down=seaice_positive_down,
    )


def wmt_by_density_bins(
    forcing,
    density=None,
    *,
    salinity=None,
    temperature=None,
    p=0,
    transform_type="raw",
    density_kind="sigma0",
    density_bins=None,
    density_min=24,
    density_max=29,
    density_step=0.1,
    time_dim=None,
    normalize_by_time=True,
    reduce_dims=None,
    preserve_dims=None,
    area=None,
    output_unit="Sv",
    area_align="auto",
    wmt_sign="positive_densification",
    lat_boundary=None,
    lon=None,
    lat=None,
    input_kind="SP_pt",
    return_density=False,
    heat_positive_down=True,
    fwf_positive_down=True,
    seaice_positive_down=True,
):
    """
    Bin water-mass transformation by density.

    ``forcing`` can either be an already computed transformation rate
    (``transform_type='raw'``) or a heat/freshwater/sea-ice forcing that should
    first be passed through the corresponding ``trans_rate_*`` function.

    If ``density`` is None, it is calculated from ``salinity`` and
    ``temperature`` using ``density_kind``. Use ``density_kind='rho'`` for bins
    around 1024-1029, or ``density_kind='sigma0'`` for bins around 24-29.
    If ``lon`` and ``lat`` are not provided, the function tries to infer them
    from the coordinates attached to ``forcing``, ``density``, ``salinity``, or
    ``temperature``.
    If ``area`` is provided, the transformation rate is multiplied by cell area
    before binning. Set ``output_unit='Sv'`` to divide the area-integrated result
    by 1e6.
    If ``lat_boundary`` is provided, only grid cells with ``lat < lat_boundary``
    are included.
    By default positive WMT means densification, i.e. transformation toward
    higher density. Set ``wmt_sign='positive_lightening'`` for the opposite
    convention.

    Returns a DataArray with a ``density`` dimension. Any dimensions not reduced
    remain on the output, which allows this function to preserve dimensions such
    as ``block`` when requested. If ``return_density=True``, returns a Dataset
    containing both ``wmt_by_density`` and ``density_field``.
    """
    centers, step = _density_centers(density_bins, density_min, density_max, density_step)
    lon, lat = _infer_lon_lat(forcing, density, salinity, temperature, lon=lon, lat=lat)
    forcing = _apply_lat_boundary(forcing, lat, lat_boundary)
    density = _apply_lat_boundary(density, lat, lat_boundary)
    salinity = _apply_lat_boundary(salinity, lat, lat_boundary)
    temperature = _apply_lat_boundary(temperature, lat, lat_boundary)
    if density is None:
        if salinity is None or temperature is None:
            raise ValueError("salinity and temperature are required when density is None")
        density = density_from_salinity_temperature(
            salinity,
            temperature,
            p=p,
            lon=lon,
            lat=lat,
            input_kind=input_kind,
            density_kind=density_kind,
        )

    transform = _transformation_from_forcing(
        forcing,
        salinity,
        temperature,
        p,
        transform_type=transform_type,
        lon=lon,
        lat=lat,
        input_kind=input_kind,
        heat_positive_down=heat_positive_down,
        fwf_positive_down=fwf_positive_down,
        seaice_positive_down=seaice_positive_down,
    )
    transform, density = xr.align(transform, density, join="inner")
    sign_mode = _normalize_wmt_sign(wmt_sign)
    sign_factor = 1 if sign_mode == "positive_densification" else -1
    wmt_rate = sign_factor * transform / step
    wmt_rate, result_units = _apply_area_and_output_unit(wmt_rate, area, output_unit, area_align)

    if reduce_dims is None:
        preserved = set(preserve_dims or ())
        reduce_dims = [dim for dim in wmt_rate.dims if dim not in preserved]
    else:
        reduce_dims = list(reduce_dims)

    inferred_time_dim = None
    if normalize_by_time:
        inferred_time_dim = _infer_time_dim(wmt_rate, time_dim=time_dim)

    pieces = []
    for center in centers:
        lower = center - step / 2
        upper = center + step / 2
        mask = (density >= lower) & (density < upper)
        binned = wmt_rate.where(mask).sum(dim=reduce_dims, skipna=True)
        if normalize_by_time:
            binned = binned / wmt_rate.sizes[inferred_time_dim]
        pieces.append(binned)

    out = xr.concat(pieces, dim=xr.IndexVariable("density", centers))
    out.name = "wmt_by_density"
    out.attrs.update(
        {
            "long_name": "Water-mass transformation binned by density",
            "density_bin_width": step,
            "density_bin_units": density.attrs.get("units", "kg m-3") if hasattr(density, "attrs") else "kg m-3",
            "density_kind": density_kind,
            "transform_type": _normalize_transform_type(transform_type),
            "normalized_by_time": bool(normalize_by_time),
            "units": result_units,
            "output_unit": "native" if output_unit is None else output_unit,
            "area_weighted": bool(area is not None),
            "wmt_sign": sign_mode,
        }
    )
    if not return_density:
        return out

    density_field = density.copy()
    density_field.name = "density_field"
    return xr.Dataset(
        {
            "wmt_by_density": out,
            "density_field": density_field,
        }
    )


def _convert_forcing_unit(forcing, unit):
    normalized = str(unit).lower()
    if normalized == "sv":
        out = forcing / 1e6
        out.attrs["units"] = "Sv"
        return out
    if normalized == "pg":
        seconds_per_year = 365.25 * 24 * 3600
        out = forcing / 1e12 * seconds_per_year
        out.attrs["units"] = "Pg yr-1"
        return out
    if normalized in {"none", "native", "raw"}:
        return forcing
    raise ValueError("unit must be one of: 'Sv', 'Pg', or 'native'")


def _isel_last_dims(arr, *indexers):
    dims = arr.dims[-len(indexers) :]
    return arr.isel({dim: indexer for dim, indexer in zip(dims, indexers)})


def _apply_blocked_model_fixes(model, forcing, density, temperature, salinity, forcing_name, apply_model_fixes=True):
    if not apply_model_fixes or not model:
        return forcing, density, temperature, salinity

    if "CMCC" in model:
        density = _isel_last_dims(density, slice(None, -1), slice(1, -1))
        temperature = _isel_last_dims(temperature, slice(None, -1), slice(1, -1))
        salinity = _isel_last_dims(salinity, slice(None, -1), slice(1, -1))

    if "NorESM" in model and "sid" in forcing_name:
        density = _isel_last_dims(density, slice(None, -1), slice(None))
        temperature = _isel_last_dims(temperature, slice(None, -1), slice(None))
        salinity = _isel_last_dims(salinity, slice(None, -1), slice(None))

    if "FGOALS" in model and "j" in density.dims:
        density = density.isel(j=slice(None, None, -1))
        temperature = temperature.isel(j=slice(None, None, -1))
        salinity = salinity.isel(j=slice(None, None, -1))

    if "CAS-ESM" in model:
        last_dim = forcing.dims[-1]
        forcing = forcing.roll({last_dim: 1}, roll_coords=True)

    return forcing, density, temperature, salinity


def blocked_wmt_by_density_bins(
    data,
    forcing_name="sidd_weighted_monthly",
    *,
    density_name="density_monthly",
    temperature_name="tos_monthly",
    salinity_name="sos_monthly",
    area_name="areacello",
    model=None,
    density_min=24,
    density_max=29,
    density_step=0.1,
    density_kind="sigma0",
    density_bins=None,
    unit="Sv",
    area=None,
    output_unit="Sv",
    area_align="auto",
    wmt_sign="positive_densification",
    transform_type="seaice",
    block_dim="block",
    time_dim=None,
    lat_boundary=-45,
    apply_model_fixes=True,
    input_kind="SP_pt",
    p=0,
    lon=None,
    lat=None,
    return_density=False,
    heat_positive_down=True,
    fwf_positive_down=True,
    seaice_positive_down=True,
):
    """
    Compute WMT(density) for data already grouped by a block dimension.

    ``data`` should usually be ``data_dict[model]`` and contain forcing,
    temperature, and salinity DataArrays. If ``density_name`` exists in
    ``data``, that density field is used for binning. Otherwise density is
    calculated from salinity and temperature. The WMT output has dimensions
    ``(block_dim, density)``.
    """
    forcing = data[forcing_name]
    temperature = data[temperature_name]
    salinity = data[salinity_name]
    if area is None and area_name is not None and area_name in data:
        area = data[area_name]
    lon, lat = _infer_lon_lat(forcing, salinity, temperature, lon=lon, lat=lat)
    if density_name is not None and density_name in data:
        density = data[density_name]
        density_source = density_name
    else:
        density = density_from_salinity_temperature(
            salinity,
            temperature,
            p=p,
            lon=lon,
            lat=lat,
            input_kind=input_kind,
            density_kind=density_kind,
        )
        density_source = f"computed:{density_kind}"
    lon, lat = _infer_lon_lat(forcing, density, salinity, temperature, lon=lon, lat=lat)

    forcing = _convert_forcing_unit(forcing, unit)
    time_dim = _infer_time_dim(forcing, time_dim=time_dim)

    forcing, density, temperature, salinity = _apply_blocked_model_fixes(
        model,
        forcing,
        density,
        temperature,
        salinity,
        forcing_name,
        apply_model_fixes=apply_model_fixes,
    )

    for arr, name in (
        (forcing, forcing_name),
        (density, density_source),
        (temperature, temperature_name),
        (salinity, salinity_name),
    ):
        if block_dim not in arr.dims:
            raise ValueError(f"{name} missing block dim {block_dim!r}. dims={arr.dims}")

    forcing, density, temperature, salinity = xr.align(forcing, density, temperature, salinity, join="inner")
    block_vals = forcing[block_dim].values
    results = []
    for block_value in block_vals:
        selector = {block_dim: block_value}
        results.append(
            wmt_by_density_bins(
                forcing.sel(selector),
                density.sel(selector),
                salinity=salinity.sel(selector),
                temperature=temperature.sel(selector),
                p=p,
                transform_type=transform_type,
                density_kind=density_kind,
                density_bins=density_bins,
                density_min=density_min,
                density_max=density_max,
                density_step=density_step,
                time_dim=time_dim,
                normalize_by_time=True,
                area=area.sel(selector) if isinstance(area, xr.DataArray) and block_dim in area.dims else area,
                output_unit=output_unit,
                area_align=area_align,
                wmt_sign=wmt_sign,
                lat_boundary=lat_boundary,
                lon=lon.sel(selector) if isinstance(lon, xr.DataArray) and block_dim in lon.dims else lon,
                lat=lat.sel(selector) if isinstance(lat, xr.DataArray) and block_dim in lat.dims else lat,
                input_kind=input_kind,
                heat_positive_down=heat_positive_down,
                fwf_positive_down=fwf_positive_down,
                seaice_positive_down=seaice_positive_down,
            )
        )

    out = xr.concat(results, dim=xr.IndexVariable(block_dim, block_vals))
    out.attrs.update(
        {
            "note": f"Computed per {block_dim}; normalized by n({time_dim}) within each block.",
            "source_forcing": forcing_name,
            "density_source": density_source,
            "temperature_source": temperature_name,
            "salinity_source": salinity_name,
            "unit_conversion": unit,
            "area_source": area_name if area is not None else None,
            "output_unit": output_unit,
            "wmt_sign": _normalize_wmt_sign(wmt_sign),
            "lat_boundary": lat_boundary,
        }
    )
    if not return_density:
        return out

    density_field = density.copy()
    density_field.name = "density_field"
    return xr.Dataset(
        {
            "wmt_by_density": out,
            "density_field": density_field,
        }
    )


def calculate_wmt_from_blocked(
    data_dict,
    model,
    variable_name="sidd_weighted_monthly",
    density_min=24,
    density_max=29,
    step=0.1,
    unit="Sv",
    v_type="sea ice",
    block_dim="block",
    time_dim=None,
    lat_thresh=-45,
    return_density=False,
    area_name="areacello",
    output_unit="Sv",
    wmt_sign="positive_densification",
):
    """
    Backward-compatible wrapper around ``blocked_wmt_by_density_bins``.

    Prefer calling ``blocked_wmt_by_density_bins(data_dict[model], ...)`` in new
    code. This wrapper keeps your previous notebook template usable.
    """
    return blocked_wmt_by_density_bins(
        data_dict[model],
        forcing_name=variable_name,
        model=model,
        density_min=density_min,
        density_max=density_max,
        density_step=step,
        unit=unit,
        transform_type=v_type,
        block_dim=block_dim,
        time_dim=time_dim,
        lat_boundary=lat_thresh,
        return_density=return_density,
        area_name=area_name,
        output_unit=output_unit,
        wmt_sign=wmt_sign,
    )


def smooth_along_density(da, window=3, density_dim="density", center=True, min_periods=1):
    """Smooth a WMT curve along the density coordinate."""
    if window is None or window <= 1:
        return da
    if density_dim not in da.dims:
        raise ValueError(f"density_dim {density_dim!r} not found in dims={da.dims}")
    if getattr(da.data, "chunks", None) is not None:
        da = da.chunk({density_dim: -1})
    return da.rolling({density_dim: window}, center=center, min_periods=min_periods).mean()


def shade_density_ranges(ax, x, mask, *, color="lightgray", alpha=0.3, zorder=0):
    """
    Shade continuous x-ranges where mask is True.

    Parameters
    ----------
    ax
        Matplotlib axes.
    x
        1D density coordinate.
    mask
        Boolean array with the same length as x.
    """
    x = np.asarray(x)
    mask = np.asarray(mask, dtype=bool)
    if x.size != mask.size:
        raise ValueError("x and mask must have the same length")
    if x.size == 0:
        return ax

    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    for start, stop in edges.reshape(-1, 2):
        left = x[start]
        right = x[stop - 1]
        if stop - start == 1 and x.size > 1:
            dx = np.nanmedian(np.diff(x))
            left = left - dx / 2
            right = right + dx / 2
        ax.axvspan(left, right, color=color, alpha=alpha, zorder=zorder, linewidth=0)
    return ax


def _as_density_curve(da, density_dim="density"):
    if isinstance(da, xr.Dataset):
        if "wmt_by_density" not in da:
            raise ValueError("Dataset input must contain 'wmt_by_density'")
        da = da["wmt_by_density"]
    if density_dim not in da.dims:
        raise ValueError(f"density_dim {density_dim!r} not found in dims={da.dims}")
    reduce_dims = [dim for dim in da.dims if dim != density_dim]
    if reduce_dims:
        da = da.mean(dim=reduce_dims, skipna=True)
    return da


def _plot_shade_condition(shade, shade_condition, heat, water, total, density_range, density_dim):
    if shade_condition is not None:
        return shade_condition(heat, water, total)
    if shade is None:
        return None

    x = heat[density_dim]
    if shade == "heat_positive":
        mask = heat > 0
    elif shade == "water_negative":
        mask = water < 0
    elif shade == "opposing":
        mask = heat * water < 0
    else:
        raise ValueError("shade must be one of: None, 'heat_positive', 'water_negative', or 'opposing'")

    if density_range is not None:
        lo, hi = density_range
        mask = mask & (x >= lo) & (x <= hi)
    return mask


def plot_wmt_components(
    wmt_heat,
    wmt_water,
    wmt_total=None,
    *,
    ax=None,
    density_dim="density",
    smooth=True,
    smooth_window=3,
    density_range=None,
    ylim=None,
    shade=None,
    shade_condition=None,
    shade_density_range=None,
    colors=None,
    labels=None,
    title=None,
    xlabel=None,
    ylabel="WMT (Sv)",
    grid=True,
    show_legend=False,
):
    """
    Plot total, heat-driven, and freshwater-driven WMT for one model.

    Returns
    -------
    ax
        The matplotlib axes used for plotting.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.5))

    heat = _as_density_curve(wmt_heat, density_dim=density_dim)
    water = _as_density_curve(wmt_water, density_dim=density_dim)
    total = heat + water if wmt_total is None else _as_density_curve(wmt_total, density_dim=density_dim)
    total, heat, water = xr.align(total, heat, water, join="inner")

    if smooth:
        heat = smooth_along_density(heat, window=smooth_window, density_dim=density_dim)
        water = smooth_along_density(water, window=smooth_window, density_dim=density_dim)
        total = smooth_along_density(total, window=smooth_window, density_dim=density_dim)

    x = total[density_dim].values
    shade_mask = _plot_shade_condition(shade, shade_condition, heat, water, total, shade_density_range, density_dim)
    if shade_mask is not None:
        shade_density_ranges(ax, x, shade_mask.values if hasattr(shade_mask, "values") else shade_mask)

    colors = colors or {"total": "k", "heat": "tab:red", "water": "tab:blue"}
    labels = labels or {"total": "Total", "heat": "Heat-driven", "water": "Freshwater-driven"}

    ax.plot(x, total.values, color=colors["total"], lw=1.8, label=labels["total"], zorder=3)
    ax.plot(x, heat.values, color=colors["heat"], lw=1.4, label=labels["heat"], zorder=3)
    ax.plot(x, water.values, color=colors["water"], lw=1.4, label=labels["water"], zorder=3)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)

    if density_range is not None:
        ax.set_xlim(*density_range)
    if ylim is not None:
        ax.set_ylim(*ylim if isinstance(ylim, tuple) else (-ylim, ylim))
    if grid:
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if show_legend:
        ax.legend(frameon=True)
    return ax


def _global_wmt_ylim(data_dict, model_list, heat_key, water_key, total_key, density_dim, smooth, smooth_window):
    ymin, ymax = [], []
    for model in model_list:
        heat = _as_density_curve(data_dict[model][heat_key], density_dim=density_dim)
        water = _as_density_curve(data_dict[model][water_key], density_dim=density_dim)
        total = heat + water if total_key is None else _as_density_curve(data_dict[model][total_key], density_dim=density_dim)
        total, heat, water = xr.align(total, heat, water, join="inner")
        if smooth:
            heat = smooth_along_density(heat, window=smooth_window, density_dim=density_dim)
            water = smooth_along_density(water, window=smooth_window, density_dim=density_dim)
            total = smooth_along_density(total, window=smooth_window, density_dim=density_dim)
        ymin.append(np.nanmin([float(total.min()), float(heat.min()), float(water.min())]))
        ymax.append(np.nanmax([float(total.max()), float(heat.max()), float(water.max())]))
    lim = max(abs(np.nanmin(ymin)), abs(np.nanmax(ymax)))
    return (-lim, lim)


def plot_wmt_model_grid(
    data_dict,
    model_list,
    *,
    heat_key="wmt_hfds_block_mean",
    water_key="wmt_wfo_block_mean",
    total_key=None,
    ncols=4,
    density_dim="density",
    density_range=(24, 29),
    xlabel=None,
    ylabel="WMT (Sv)",
    smooth=True,
    smooth_window=3,
    ylim="symmetric",
    shade="heat_positive",
    shade_condition=None,
    shade_density_range=None,
    figsize_per_panel=(3.5, 2.8),
    legend=True,
    legend_loc="lower left",
    legend_bbox=(0.8, 0.15),
    savepath=None,
    dpi=300,
):
    """
    Plot WMT component curves for multiple models in a subplot grid.

    ``data_dict[model][heat_key]`` and ``data_dict[model][water_key]`` should be
    DataArrays with a ``density`` dimension. If they still contain a block or
    time dimension, they are averaged before plotting.
    """
    import math
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    n_models = len(model_list)
    nrows = math.ceil(n_models / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )
    flat_axes = axes.flatten()

    if ylim == "symmetric":
        resolved_ylim = _global_wmt_ylim(data_dict, model_list, heat_key, water_key, total_key, density_dim, smooth, smooth_window)
    elif ylim is None:
        resolved_ylim = None
    else:
        resolved_ylim = ylim

    xlabel = xlabel or "Potential density anomaly, sigma0 (kg m$^{-3}$)"
    shade_density_range = shade_density_range if shade_density_range is not None else density_range

    for idx, model in enumerate(model_list):
        ax = flat_axes[idx]
        heat = data_dict[model][heat_key]
        water = data_dict[model][water_key]
        total = None if total_key is None else data_dict[model][total_key]
        plot_wmt_components(
            heat,
            water,
            total,
            ax=ax,
            density_dim=density_dim,
            smooth=smooth,
            smooth_window=smooth_window,
            density_range=density_range,
            ylim=resolved_ylim,
            shade=shade,
            shade_condition=shade_condition,
            shade_density_range=shade_density_range,
            title=model,
            xlabel=xlabel if idx >= n_models - ncols else None,
            ylabel=ylabel if idx % ncols == 0 else "",
            show_legend=False,
        )
        if idx % ncols != 0:
            ax.set_yticklabels([])
        ax.tick_params(axis="both", labelsize=8)
        ax.tick_params(axis="x", labelrotation=45)

    for idx in range(n_models, len(flat_axes)):
        flat_axes[idx].set_visible(False)

    if legend:
        handles = [
            Line2D([0], [0], color="k", lw=1.8, label="Total"),
            Line2D([0], [0], color="tab:red", lw=1.4, label="Heat-driven"),
            Line2D([0], [0], color="tab:blue", lw=1.4, label="Freshwater-driven"),
        ]
        if shade is not None or shade_condition is not None:
            handles.append(Patch(facecolor="lightgray", alpha=0.3, label="Shaded density range"))
        fig.legend(
            handles=handles,
            loc=legend_loc,
            bbox_to_anchor=legend_bbox,
            ncol=1,
            fontsize=12,
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.9,
        )

    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
    return fig, axes


def load_data_in_range(
    start_year: int,
    end_year: int,
    file_pattern: str,
    *,
    step: int = 10,
    chunks: Optional[dict] = None,
    combine: str = "by_coords",
):
    """
    Load files whose names contain year ranges and subset to the requested years.

    ``file_pattern`` must contain two replacement fields, for example
    ``'/path/var_{:04d}-{:04d}.nc'``.
    """
    start_year_range = (start_year // step) * step
    end_year_range = ((end_year // step) + 1) * step
    paths = [
        file_pattern.format(year_range, year_range + step - 1)
        for year_range in range(start_year_range, end_year_range, step)
    ]
    missing = [path for path in paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing input files: {missing}")

    ds = xr.open_mfdataset(paths, combine=combine, chunks=chunks)
    return ds.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))


__all__ = [
    "CP_SEAWATER",
    "GRAVITY",
    "SEA_ICE_SALINITY",
    "to_teos10",
    "thermodynamic_coefficients",
    "density_from_salinity_temperature",
    "buoyancy_flux",
    "buoyancy_flux_heat",
    "buoyancy_flux_water",
    "buoyancy_flux_water_to_heat",
    "trans_rate_heat",
    "trans_rate_water",
    "trans_rate_seaice",
    "trans_rate_mix_sal",
    "get_lat_lon_coords",
    "align_area_to_data",
    "wmt_by_density_bins",
    "blocked_wmt_by_density_bins",
    "calculate_wmt_from_blocked",
    "smooth_along_density",
    "shade_density_ranges",
    "plot_wmt_components",
    "plot_wmt_model_grid",
    "load_data_in_range",
]
