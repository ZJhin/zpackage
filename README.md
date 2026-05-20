# zpackage

`zpackage` is a small personal Python toolbox for ocean and climate data analysis.
The repository currently contains utilities for water-mass diagnostics, color maps,
and CMIP6 data selection. The most active workflow is `Ztake`, a helper for working
with NCI `intake-esm` catalogs and ESGF online data.

> Note: the refactored workflow is currently in `ztake_refactored.py`. It is kept
> separate from the original `ztake.py` so the old workflow remains untouched while
> the new API is tested.

## Repository Layout

| File | Purpose |
| --- | --- |
| `ztake.py` | Original Ztake implementation. |
| `ztake_refactored.py` | Refactored Ztake workflow with multi-catalog support, model/member/grid inspection, local opening, online ESGF opening, and local-vs-online comparison. |
| `wmt.py` | Water-mass transformation utilities. |
| `zclef.py`, `zclef_v2.py` | Climate/ocean analysis helpers. |
| `colormap.py` | Custom color-map helpers. |
| `Sandbox.ipynb` | Working notebook for experiments and examples. |

## Environment

The examples below are designed for NCI Gadi with an `analysis3` environment:

```bash
module use /g/data/xp65/public/modules
module load conda/analysis3-26.04
```

Python dependencies used by `ztake_refactored.py` include:

```python
intake
intake_esm
xarray
pandas
numpy
requests
dask
netCDF4
cftime
```

## Basic Import

From the repository directory:

```python
from ztake_refactored import Ztake
import intake
```

If the repository is not on `PYTHONPATH`, add it first:

```python
import sys
sys.path.insert(0, "/g/data/jk72/zc0441/zpackage")

from ztake_refactored import Ztake
```

## 1. Open One Or More Intake Catalogs

`Ztake` can use one catalog, a list of catalogs, or a named dictionary of catalogs.
Using named catalogs is recommended because the output tables show where each
dataset came from.

```python
cmip6_oi10 = intake.cat.access_nri["cmip6_oi10"]
cmip6_fs38 = intake.cat.access_nri["cmip6_fs38"]

catalogs = {
    "oi10": cmip6_oi10,
    "fs38": cmip6_fs38,
}
```

This avoids creating two separate `Ztake` objects when you want to search multiple
catalogs for the same model/variable workflow.

## 2. Define Search Constraints

Use CMIP6-style fields such as `experiment_id`, `variable_id`, `member_id`,
`table_id`, and `grid_label`.

```python
constraints = {
    "experiment_id": ["historical", "piControl"],
    "variable_id": ["thetao", "so", "po4", "o2", "wmo"],
    "member_id": ["r1i1p1f1", "r2i1p1f1", "r1i1p1f2"],
    "table_id": "Omon",
}
```

List values are treated as "choose any of these". The refactored code expands
list-valued searches internally to avoid unwanted intake value-alias warnings.

## 3. Build A Ztake Object

```python
zt = Ztake(
    catalogs,
    constraints=constraints,
    prefer_members=("r1i1p1f1", "r1i1p1f2", "r2i1p1f1"),
    prefer_grids=("gn", "gr", "gr1"),
    prefer_catalogs=("oi10", "fs38"),
)
```

Preference order is only used when multiple files satisfy the same model,
experiment, and variable. You can still request a specific `member_id` or
`grid_label` later when opening data.

## 4. Inspect Available Models

Quick model list:

```python
zt.model_list()
zt.show_models()
```

Compact model table:

```python
zt.model_table()
```

All matching rows before best-row selection:

```python
zt.model_table(selected=False)
```

Which catalog contains which model:

```python
zt.models_by_catalog()
zt.model_presence()
```

## 5. Inspect Member And Grid Options

Show every available ensemble member and grid for each model:

```python
zt.member_grid_table()
```

Show options for one model:

```python
zt.options_for_model("ACCESS-ESM1-5")
```

Limit to one experiment:

```python
zt.options_for_model("ACCESS-ESM1-5", experiment_id="piControl")
zt.member_grid_table(experiment_id="historical")
```

This is the recommended step before opening data, especially when a model has
multiple ensemble members or grid labels.

## 6. Open Local NCI Data

Open one model using the preferred member/grid:

```python
ds = zt.open_model(
    "ACCESS-ESM1-5",
    experiment_id="piControl",
    variables=["thetao", "so", "po4", "o2"],
    use_cftime=True,
    parallel=False,
)
```

Open a specific ensemble member or grid:

```python
ds = zt.open_model(
    "CESM2",
    experiment_id="historical",
    variables=["thetao"],
    member_id="r2i1p1f1",
    grid_label="gn",
    time_range=("2000-01", "2000-03"),
    use_cftime=True,
    parallel=False,
)
```

Open all selected models:

```python
datasets = zt.open(
    experiment_id="historical",
    variables=["thetao", "so"],
    use_cftime=True,
    parallel=False,
)

ds_cesm2 = datasets["CESM2"]
```

When opening multiple variables with different vertical coordinates, xarray may
warn about `join='outer'`. This means coordinates such as `lev` are being aligned
across variables. You can make this explicit:

```python
ds = zt.open_model(
    "ACCESS-ESM1-5",
    experiment_id="piControl",
    variables=["thetao", "so", "wmo"],
    join="outer",
    use_cftime=True,
)
```

If you require identical coordinates across variables, use `join="exact"` and
let xarray raise an error when coordinates do not match.

## 7. Open Online ESGF Data

Use this when a dataset is not available locally or you want to test remote
OPeNDAP access.

```python
ds = zt.open_online_model(
    "CESM2",
    variable_id="spco2",
    experiment_id="piControl",
    member_id="r1i1p1f1",
    table_id="Omon",
    grid_label="gn",
    time_range=("2000-01", "2000-03"),
    keep_variable_only=True,
    local_node=True,
    parallel=False,
)
```

Important: `open_online_model()` returns an xarray dataset backed by remote URLs.
Metadata and coordinates may be opened immediately, but the full data are usually
loaded lazily. Actual array values are read when you compute, plot, save, or call
`.load()`.

To inspect the URLs first:

```python
urls = zt.online_urls_for_model(
    "CESM2",
    variable_id="spco2",
    experiment_id="piControl",
    member_id="r1i1p1f1",
    table_id="Omon",
    grid_label="gn",
)

urls[:3]
```

To inspect matching ESGF file metadata:

```python
zt.online_file_table_for_model(
    "CESM2",
    variable_id="spco2",
    experiment_id="piControl",
    member_id="r1i1p1f1",
    table_id="Omon",
    grid_label="gn",
)
```

## 8. Compare Local Catalogs With ESGF

Check whether local NCI catalog results match online ESGF records:

```python
comparison = zt.compare_with_esgf(
    mode="latest",
    verbose=True,
)

comparison["local_count"]
comparison["online_count"]
comparison["only_local"]
comparison["only_online"]
comparison["version_mismatch"]
```

Useful modes:

| Mode | Meaning |
| --- | --- |
| `latest` | Compare latest local versions with latest ESGF versions. |
| `ignore_version` | Compare dataset identity without requiring matching version labels. |
| `all_versions` | Compare all visible versions. |

If you want a file containing online-only ESGF instance IDs:

```python
zt.compare_with_esgf(
    request_ids=True,
    filename="only_online_instance_ids.txt",
)
```

## Common Notes

- `verbose=True` means "print more diagnostic information while running".
- The ambiguous cftime warning for dates such as `101-01-01` means xarray padded
  the year to `0101-01-01`. It is usually informational for CMIP control runs.
- Online ESGF datasets are lazy until values are actually computed or loaded.
- Keep `ztake_refactored.py` separate while testing. Once the API is stable, it
  can be merged back into `ztake.py` or turned into a package module.

## WMT Refactored Workflow

`wmt_refactored.py` contains a safer water-mass-transformation workflow. It keeps
the original `wmt.py` untouched, but adds TEOS-10 conversion, density-bin WMT,
area-weighted Sv output, latitude masking, and quick plotting helpers.

Basic import:

```python
import sys
sys.path.insert(0, "/g/data/jk72/zc0441/zpackage")

import wmt_refactored as wmt
```

### Compute WMT in density bins

The default density coordinate is now `sigma0`, with bins from 24 to 29 at 0.1
intervals. Positive WMT means densification, or transformation toward higher
density. Because the default output unit is `Sv`, pass cell area with
`area=areacello`.

```python
wmt_wfo = wmt.wmt_by_density_bins(
    forcing=wfo,
    density=None,
    salinity=sos,
    temperature=tos,
    area=areacello,
    transform_type="water",
    lat_boundary=-45,
)
```

For heat-driven WMT:

```python
wmt_hfds = wmt.wmt_by_density_bins(
    forcing=hfds,
    density=None,
    salinity=sos,
    temperature=tos,
    area=areacello,
    transform_type="heat",
    lat_boundary=-45,
)
```

Useful defaults:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `density_kind` | `"sigma0"` | Use potential density anomaly for bins. |
| `density_min`, `density_max` | `24`, `29` | Density-bin range. |
| `density_step` | `0.1` | Density-bin width. |
| `output_unit` | `"Sv"` | Area-integrated WMT divided by `1e6`. |
| `wmt_sign` | `"positive_densification"` | Positive values mean transformation toward higher density. |

If you want the density field returned with the binned WMT:

```python
out = wmt.wmt_by_density_bins(
    forcing=wfo,
    density=None,
    salinity=sos,
    temperature=tos,
    area=areacello,
    transform_type="water",
    lat_boundary=-45,
    return_density=True,
)

wmt_wfo = out["wmt_by_density"]
sigma0 = out["density_field"]
```

### Blocked decade workflow

If data have already been preprocessed into blocks, for example
`(block, month, j, i)`, use:

```python
wmt_wfo_block = wmt.blocked_wmt_by_density_bins(
    data_dict[model],
    forcing_name="wfo_monthly",
    density_name=None,
    salinity_name="sos_monthly",
    temperature_name="tos_monthly",
    area_name="areacello",
    transform_type="water",
    lat_boundary=-45,
)
```

The blocked helper automatically infers `time` or `month`, and it tries to read
`lat/lon` from coordinates. It does not silently regrid `areacello`; if the area
grid is incompatible with the data grid, pre-align it before calling the WMT
function.

### Quick WMT plots

Single model:

```python
ax = wmt.plot_wmt_components(
    wmt_heat=wmt_hfds,
    wmt_water=wmt_wfo,
    density_range=(24, 29),
    title=model,
)
```

Model grid:

```python
fig, axes = wmt.plot_wmt_model_grid(
    data_dict,
    model_list,
    heat_key="wmt_hfds_block_mean",
    water_key="wmt_wfo_block_mean",
    ncols=4,
    density_range=(24, 29),
    ylim="symmetric",
    savepath="wmt_binned_pi.png",
)
```

The plotting helper smooths along density, handles Dask-backed arrays by
rechunking the density dimension, and returns `fig, axes` without forcing
`plt.show()`.

## Recommended Ztake Workflow

```python
from ztake_refactored import Ztake
import intake

catalogs = {
    "oi10": intake.cat.access_nri["cmip6_oi10"],
    "fs38": intake.cat.access_nri["cmip6_fs38"],
}

constraints = {
    "experiment_id": ["historical", "piControl"],
    "variable_id": ["thetao", "so", "po4", "o2", "wmo"],
    "member_id": ["r1i1p1f1", "r2i1p1f1", "r1i1p1f2"],
    "table_id": "Omon",
}

zt = Ztake(catalogs, constraints=constraints)

zt.show_models()
zt.member_grid_table()
zt.options_for_model("ACCESS-ESM1-5", experiment_id="piControl")

ds = zt.open_model(
    "ACCESS-ESM1-5",
    experiment_id="piControl",
    variables=["thetao", "so"],
    member_id="r1i1p1f1",
    grid_label="gn",
    use_cftime=True,
    parallel=False,
)
```
