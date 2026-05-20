from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import warnings

import numpy as np
import pandas as pd
import requests
import xarray as xr


ESGF_NODES_DEFAULT: Tuple[str, ...] = (
    "https://esgf-node.llnl.gov/esg-search/search/",
    "https://esgf-data.dkrz.de/esg-search/search/",
    "https://esgf-node.ipsl.upmc.fr/esg-search/search/",
    "https://esgf.nci.org.au/esg-search/search/",
    "https://esgf-node.ornl.gov/esg-search/search/",
)

CORE_COORDS: Tuple[str, ...] = (
    "time",
    "lat",
    "latitude",
    "nav_lat",
    "TLAT",
    "yt_ocean",
    "lon",
    "longitude",
    "nav_lon",
    "TLONG",
    "xt_ocean",
    "x",
    "y",
    "i",
    "j",
)

CatalogSpec = Any


def _first(value: Any) -> Any:
    return value[0] if isinstance(value, list) and value else value


def _as_list(value: Any) -> Optional[List[Any]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _expanded_constraint_sets(
    constraints: Mapping[str, Any],
    keys: Sequence[str],
) -> List[Dict[str, Any]]:
    """
    Expand list-valued ESGF constraints into separate single-value queries.

    ESGF Solr treats repeated parameters for the same field as AND filters. Local
    intake catalogs usually treat list values as OR filters. Expanding list
    values preserves the local intake semantics for online comparison.
    """
    base = dict(constraints)
    expanded_keys = [key for key in keys if isinstance(base.get(key), (list, tuple, set))]
    if not expanded_keys:
        return [base]

    value_lists = [list(base[key]) for key in expanded_keys]
    queries: List[Dict[str, Any]] = []
    for combo in product(*value_lists):
        query = dict(base)
        for key, value in zip(expanded_keys, combo):
            query[key] = value
        queries.append(query)
    return queries


def _version_number(value: Any) -> int:
    text = str(value).strip().lower()
    if text == "latest":
        return 0
    if text.startswith(("v", "d")):
        text = text[1:]
    try:
        return int(text)
    except (TypeError, ValueError):
        return 0


def _version_label(value: Any) -> Optional[str]:
    text = str(value).strip()
    if not text or text.lower() == "latest":
        return None
    if text[0].lower() in {"v", "d"}:
        text = text[1:]
    if not text.isdigit():
        return None
    return f"v{text}"


def _normalize_constraints(constraints: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = dict(constraints)
    if "variable" in normalized and "variable_id" not in normalized:
        normalized["variable_id"] = normalized.pop("variable")
    return normalized


def _filter_frame_to_constraints(df: pd.DataFrame, constraints: Mapping[str, Any]) -> pd.DataFrame:
    """
    Remove rows introduced only by intake catalog value aliases.

    Some access_nri catalogs alias values such as thetao -> pot_temp. For Ztake,
    user constraints should be treated literally unless the user explicitly
    requested the alias value.
    """
    out = df
    for key, value in constraints.items():
        if key not in out.columns or value is None:
            continue
        allowed = set(_as_list(value) or [])
        if not allowed:
            continue
        out = out[out[key].isin(allowed)]
    return out


def _require_columns(df: pd.DataFrame, columns: Iterable[str], context: str = "dataframe") -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"{context} is missing required columns: {missing}")


@dataclass
class Ztake:
    """
    Select CMIP6 datasets from an intake-ESM catalog.

    The class chooses one preferred dataset per ``(source_id, variable_id)`` using
    member, grid, and version preferences; opens selected files with xarray; and
    can compare local catalog availability with ESGF Solr search results.
    """

    cmip6_catalog: CatalogSpec
    constraints: Mapping[str, Any]
    chunks: Mapping[str, int] = field(default_factory=lambda: {"time": 12})
    prefer_members: Sequence[str] = ("r1i1p1f1", "r1i1p1f2", "r1i1p1f3")
    prefer_grids: Sequence[str] = ("gn", "gr", "gr1")
    prefer_catalogs: Optional[Sequence[str]] = None

    filtered_ds: Any = field(init=False, repr=False)
    _constraints: Dict[str, Any] = field(init=False, repr=False)
    _catalogs: Dict[str, Any] = field(init=False, repr=False)
    _df: pd.DataFrame = field(init=False, repr=False)
    _best_df: pd.DataFrame = field(init=False, repr=False)
    _model_list: List[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._constraints = _normalize_constraints(self.constraints)
        self._catalogs = self._normalize_catalogs(self.cmip6_catalog)
        self.filtered_ds, self._df = self._search_catalogs(self._catalogs, self._constraints)

        if self._df.empty:
            raise ValueError(f"No catalog rows matched constraints: {self._constraints}")

        _require_columns(
            self._df,
            ["source_id", "variable_id", "member_id", "grid_label", "version"],
            context="catalog search result",
        )
        self._df = self._prepare_dataframe(self._df)
        self._best_df = self._select_best_rows(self._df)
        self._model_list = sorted(self._best_df["source_id"].unique().tolist())

    @staticmethod
    def _normalize_catalogs(catalogs: CatalogSpec) -> Dict[str, Any]:
        """
        Accept one catalog, a sequence of catalogs, or a name -> catalog mapping.

        Examples
        --------
        Ztake(cmip6, constraints)
        Ztake([cmip6, cmip6_fs38], constraints)
        Ztake({"oi10": cmip6, "fs38": cmip6_fs38}, constraints)
        """
        if isinstance(catalogs, Mapping):
            normalized = {str(name): catalog for name, catalog in catalogs.items()}
        elif isinstance(catalogs, Sequence) and not isinstance(catalogs, (str, bytes)):
            normalized = {f"catalog_{idx}": catalog for idx, catalog in enumerate(catalogs)}
        else:
            normalized = {"catalog_0": catalogs}

        if not normalized:
            raise ValueError("At least one intake catalog is required")
        return normalized

    @staticmethod
    def _search_catalogs(catalogs: Mapping[str, Any], constraints: Mapping[str, Any]) -> Tuple[Dict[str, Any], pd.DataFrame]:
        search_results: Dict[str, Any] = {}
        frames: List[pd.DataFrame] = []
        errors: Dict[str, str] = {}
        query_keys = tuple(constraints.keys())
        query_constraints = _expanded_constraint_sets(constraints, query_keys)

        for name, catalog in catalogs.items():
            for query in query_constraints:
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message="Value aliasing:.*", category=UserWarning)
                        result = catalog.search(**query)
                    search_results[name] = result
                    frame = result.df.copy()
                except Exception as exc:
                    errors[name] = f"{type(exc).__name__}: {exc}"
                    continue

                if frame.empty:
                    continue

                frame = _filter_frame_to_constraints(frame, query)
                if frame.empty:
                    continue
                frame["catalog_label"] = name
                frames.append(frame)

        if frames:
            merged = pd.concat(frames, ignore_index=True, sort=False)
            return search_results, merged.drop_duplicates().reset_index(drop=True)

        detail = f" Search errors: {errors}" if errors else ""
        return search_results, pd.DataFrame()

    @staticmethod
    def _rank_by_preference(values: pd.Series, preferences: Sequence[str]) -> pd.Series:
        rank_map = {value: idx for idx, value in enumerate(preferences)}
        return values.map(rank_map).fillna(len(preferences) + 100).astype(int)

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "path" not in out.columns:
            if "uri" not in out.columns:
                raise KeyError("catalog search result needs either a 'path' or 'uri' column")
            out["path"] = out["uri"]

        out["path"] = out["path"].astype(str)
        out["version_num"] = out["version"].apply(_version_number)
        out["member_rank"] = self._rank_by_preference(out["member_id"], self.prefer_members)
        out["grid_rank"] = self._rank_by_preference(out["grid_label"], self.prefer_grids)
        catalog_preferences = self.prefer_catalogs or tuple(self._catalogs.keys())
        out["catalog_rank"] = self._rank_by_preference(out["catalog_label"], catalog_preferences)
        return out

    @staticmethod
    def _select_best_rows(df: pd.DataFrame) -> pd.DataFrame:
        sort_cols = ["member_rank", "grid_rank", "version_num", "catalog_rank", "path"]
        ascending = [True, True, False, True, True]
        return (
            df.sort_values(sort_cols, ascending=ascending, kind="mergesort")
            .groupby(["source_id", "experiment_id", "variable_id"], as_index=False, group_keys=False)
            .head(1)
            .reset_index(drop=True)
        )

    def _paths_for_choice(self, row: pd.Series) -> List[str]:
        mask = (
            (self._df["source_id"] == row["source_id"])
            & (self._df["experiment_id"] == row["experiment_id"])
            & (self._df["variable_id"] == row["variable_id"])
            & (self._df["member_id"] == row["member_id"])
            & (self._df["grid_label"] == row["grid_label"])
            & (self._df["version"] == row["version"])
        )
        paths = sorted(self._df.loc[mask, "path"].astype(str).tolist())
        if not paths:
            raise FileNotFoundError(
                "No files found for selected dataset "
                f"{row['source_id']} {row['variable_id']} {row['member_id']} "
                f"{row['grid_label']} {row['version']}"
            )
        return paths

    @staticmethod
    def _preprocess_dataset(ds: xr.Dataset, variable: str, keep_coords: Sequence[str]) -> xr.Dataset:
        keep = [variable]
        keep.extend(coord for coord in keep_coords if coord in ds.variables)
        existing = sorted({name for name in keep if name in ds.variables})
        if variable not in existing:
            raise KeyError(f"Variable {variable!r} was not found in opened dataset")
        return ds[existing]

    @property
    def df(self) -> pd.DataFrame:
        """All catalog rows matching the original constraints."""
        return self._df.copy()

    @property
    def best_per_model_variable(self) -> pd.DataFrame:
        """One selected row per ``(source_id, experiment_id, variable_id)``."""
        columns = [
            "catalog_label",
            "source_id",
            "experiment_id",
            "variable_id",
            "member_id",
            "grid_label",
            "version",
            "version_num",
            "path",
        ]
        return self._best_df.loc[:, [column for column in columns if column in self._best_df]].copy()

    def models(self) -> List[str]:
        return list(self._model_list)

    def models_all(self) -> List[str]:
        return sorted(self._df["source_id"].unique().tolist())

    def models_best(self) -> List[str]:
        return self.models()

    def model_list(self, selected: bool = True) -> List[str]:
        """
        Return model names as a plain list.

        Parameters
        ----------
        selected
            If True, return models after best-row selection. If False, return all
            models present in the combined catalog search result.
        """
        return self.models_best() if selected else self.models_all()

    def show_models(self, selected: bool = True, numbered: bool = True) -> List[str]:
        """
        Print model names and return the same list.

        This is notebook-friendly when you want a quick visual checklist without
        building a DataFrame manually.
        """
        models = self.model_list(selected=selected)
        for idx, model in enumerate(models, start=1):
            prefix = f"{idx:02d}. " if numbered else ""
            print(f"{prefix}{model}")
        return models

    def model_table(self, selected: bool = True) -> pd.DataFrame:
        """
        Return a compact table of available models.

        The selected table shows one row per chosen model/variable combination.
        The all table summarizes every matching catalog row before best-row
        selection.
        """
        if selected:
            cols = [
                "catalog_label",
                "source_id",
                "experiment_id",
                "variable_id",
                "member_id",
                "grid_label",
                "version",
                "path",
            ]
            return self._best_df.loc[:, [col for col in cols if col in self._best_df]].sort_values(
                ["source_id", "experiment_id", "variable_id"]
            ).reset_index(drop=True)

        return (
            self._df.groupby(["source_id", "experiment_id", "catalog_label"], as_index=False)
            .agg(
                n_rows=("source_id", "size"),
                variables=("variable_id", lambda x: ",".join(sorted(pd.unique(x)))),
                members=("member_id", lambda x: ",".join(sorted(pd.unique(x)))),
                grids=("grid_label", lambda x: ",".join(sorted(pd.unique(x)))),
                newest_version=("version_num", "max"),
            )
            .sort_values(["source_id", "experiment_id", "catalog_label"])
            .reset_index(drop=True)
        )

    def models_by_catalog(self, selected: bool = False) -> Dict[str, List[str]]:
        """
        Return ``catalog_label -> model list``.

        By default this uses all matching catalog rows, which is useful for
        checking where models exist before the best selection is applied.
        """
        rows = self._best_df if selected else self._df
        return {
            str(catalog): sorted(group["source_id"].unique().tolist())
            for catalog, group in rows.groupby("catalog_label")
        }

    def model_presence(self) -> pd.DataFrame:
        """
        Return a model-by-catalog presence matrix.

        Values are row counts, so any value greater than zero means that model
        exists in that catalog for the current constraints.
        """
        table = pd.crosstab(self._df["source_id"], self._df["catalog_label"])
        return table.sort_index(axis=0).sort_index(axis=1)

    def model_options(self, selected: bool = False) -> pd.DataFrame:
        """
        Summarize available members, grids, versions, and catalogs per model.

        Parameters
        ----------
        selected
            If True, summarize only the best selected rows. If False, summarize
            all matching rows before best-row selection.
        """
        rows = self._best_df if selected else self._df
        return (
            rows.groupby(["source_id", "experiment_id"], as_index=False)
            .agg(
                n_rows=("source_id", "size"),
                variables=("variable_id", lambda x: ",".join(sorted(pd.unique(x)))),
                catalogs=("catalog_label", lambda x: ",".join(sorted(pd.unique(x)))),
                member_ids=("member_id", lambda x: ",".join(sorted(pd.unique(x)))),
                grid_labels=("grid_label", lambda x: ",".join(sorted(pd.unique(x)))),
                versions=("version", lambda x: ",".join(sorted(pd.unique(x)))),
            )
            .sort_values(["source_id", "experiment_id"])
            .reset_index(drop=True)
        )

    def options_for_model(
        self,
        model: str,
        experiment_id: Optional[str] = None,
        selected: bool = False,
    ) -> pd.DataFrame:
        """
        Return all catalog/member/grid/version combinations for one model.
        """
        rows = self._best_df if selected else self._df
        rows = rows[rows["source_id"] == model]
        if experiment_id is not None:
            rows = rows[rows["experiment_id"] == experiment_id]
        if rows.empty:
            raise KeyError(f"Model {model!r} not found. Available models: {self.models_all()}")

        cols = [
            "catalog_label",
            "source_id",
            "experiment_id",
            "variable_id",
            "member_id",
            "grid_label",
            "version",
            "path",
        ]
        return (
            rows.loc[:, [col for col in cols if col in rows]]
            .drop_duplicates()
            .sort_values(["experiment_id", "variable_id", "member_id", "grid_label", "version", "catalog_label"])
            .reset_index(drop=True)
        )

    def member_grid_table(
        self,
        experiment_id: Optional[str] = None,
        selected: bool = False,
    ) -> pd.DataFrame:
        """
        Return one row per model/member/grid combination.

        This is useful when you want to scan which ensemble members and grids are
        available without listing every file path.
        """
        rows = self._best_df if selected else self._df
        if experiment_id is not None:
            rows = rows[rows["experiment_id"] == experiment_id]
        return (
            rows.groupby(["source_id", "experiment_id", "member_id", "grid_label"], as_index=False)
            .agg(
                n_rows=("source_id", "size"),
                variables=("variable_id", lambda x: ",".join(sorted(pd.unique(x)))),
                catalogs=("catalog_label", lambda x: ",".join(sorted(pd.unique(x)))),
                versions=("version", lambda x: ",".join(sorted(pd.unique(x)))),
            )
            .sort_values(["source_id", "experiment_id", "member_id", "grid_label"])
            .reset_index(drop=True)
        )

    def variables_for(self, model: str, experiment_id: Optional[str] = None) -> List[str]:
        rows = self._best_df[self._best_df["source_id"] == model]
        if experiment_id is not None:
            rows = rows[rows["experiment_id"] == experiment_id]
        if rows.empty:
            raise KeyError(f"Model {model!r} not found. Available models: {self.models()}")
        return sorted(rows["variable_id"].unique().tolist())

    def info(self) -> pd.DataFrame:
        return (
            self._best_df.groupby(["source_id", "experiment_id"], as_index=False)
            .agg(
                n_vars=("variable_id", "nunique"),
                variables=("variable_id", lambda x: ",".join(sorted(pd.unique(x)))),
                catalogs=("catalog_label", lambda x: ",".join(sorted(pd.unique(x)))),
                members=("member_id", lambda x: ",".join(sorted(pd.unique(x)))),
                grids=("grid_label", lambda x: ",".join(sorted(pd.unique(x)))),
                newest_version=("version_num", "max"),
            )
            .sort_values(["source_id", "experiment_id"])
            .reset_index(drop=True)
        )

    def open(
        self,
        variables: Optional[Iterable[str]] = None,
        experiment_id: Optional[str] = None,
        member_id: Optional[str] = None,
        grid_label: Optional[str] = None,
        time_range: Optional[Tuple[str, str]] = None,
        engine: Optional[str] = None,
        decode_times: bool = True,
        use_cftime: Optional[bool] = None,
        drop_conflicts: bool = True,
        join: str = "outer",
        keep_coords: Sequence[str] = CORE_COORDS,
        parallel: bool = True,
    ) -> Dict[str, xr.Dataset]:
        """Open selected datasets for all matching models."""
        selected = self._select_experiment(self._best_df, experiment_id)
        selected = self._select_variables(selected, variables)
        return {
            key: self._open_rows(
                rows,
                time_range=time_range,
                engine=engine,
                decode_times=decode_times,
                use_cftime=use_cftime,
                drop_conflicts=drop_conflicts,
                join=join,
                keep_coords=keep_coords,
                parallel=parallel,
            )
            for key, rows in self._group_open_rows(selected)
        }

    def open_model(
        self,
        model: str,
        variables: Optional[Iterable[str]] = None,
        experiment_id: Optional[str] = None,
        member_id: Optional[str] = None,
        grid_label: Optional[str] = None,
        time_range: Optional[Tuple[str, str]] = None,
        engine: Optional[str] = None,
        decode_times: bool = True,
        use_cftime: Optional[bool] = None,
        drop_conflicts: bool = True,
        join: str = "outer",
        keep_coords: Sequence[str] = CORE_COORDS,
        parallel: bool = True,
    ) -> xr.Dataset:
        """Open selected datasets for one model."""
        if model not in self._model_list:
            raise KeyError(f"Model {model!r} not found. Available models: {self._model_list}")

        if member_id is not None or grid_label is not None:
            selected = self._select_rows_for_requested_member_grid(
                model=model,
                experiment_id=experiment_id,
                variables=variables,
                member_id=member_id,
                grid_label=grid_label,
            )
        else:
            selected = self._best_df[self._best_df["source_id"] == model]
            selected = self._select_experiment(selected, experiment_id, model=model)
            selected = self._select_variables(selected, variables, model=model)
        ds = self._open_rows(
            selected,
            time_range=time_range,
            engine=engine,
            decode_times=decode_times,
            use_cftime=use_cftime,
            drop_conflicts=drop_conflicts,
            join=join,
            keep_coords=keep_coords,
            parallel=parallel,
        )
        return ds.assign_attrs(self._selection_attrs(model, selected))

    def online_urls_for_model(
        self,
        model: str,
        variable_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        member_id: Optional[str] = None,
        table_id: Optional[str] = None,
        grid_label: Optional[str] = None,
        activity_id: Optional[str] = None,
        institution_id: Optional[str] = None,
        files_type: str = "OPENDAP",
        server: str = "https://esgf-node.llnl.gov/esg-search/search",
        local_node: bool = True,
        latest: bool = True,
        limit: int = 10000,
        timeout: int = 60,
        deduplicate: bool = True,
        preferred_node: Optional[str] = None,
        fallback_distributed: bool = True,
        verbose: bool = False,
    ) -> List[str]:
        """
        Find online ESGF file URLs for a model using the current Ztake selection.

        If member/grid/table/variable are not provided, the method uses the best
        selected row for that model. Returned URLs are suitable for xarray
        OPeNDAP access when ``files_type="OPENDAP"``.
        """
        search = self._online_search_from_selection(
            model=model,
            variable_id=variable_id,
            experiment_id=experiment_id,
            member_id=member_id,
            table_id=table_id,
            grid_label=grid_label,
            activity_id=activity_id,
            institution_id=institution_id,
        )
        urls = self.esgf_file_urls(
            server=server,
            files_type=files_type,
            local_node=local_node,
            latest=latest,
            limit=limit,
            timeout=timeout,
            verbose=verbose,
            **search,
        )
        if not urls and local_node and fallback_distributed:
            urls = self.esgf_file_urls(
                server=server,
                files_type=files_type,
                local_node=False,
                latest=latest,
                limit=limit,
                timeout=timeout,
                verbose=verbose,
                **search,
            )
        if deduplicate:
            urls = self.dedup_urls_by_timeslice(urls, preferred_node=preferred_node)
        if not urls:
            raise FileNotFoundError(f"No online {files_type} URLs found for search: {search}")
        return urls

    def open_online_model(
        self,
        model: str,
        variable_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        member_id: Optional[str] = None,
        table_id: Optional[str] = None,
        grid_label: Optional[str] = None,
        activity_id: Optional[str] = None,
        institution_id: Optional[str] = None,
        time_range: Optional[Tuple[str, str]] = None,
        files_type: str = "OPENDAP",
        server: str = "https://esgf-node.llnl.gov/esg-search/search",
        local_node: bool = True,
        latest: bool = True,
        limit: int = 10000,
        timeout: int = 60,
        deduplicate: bool = True,
        preferred_node: Optional[str] = None,
        fallback_distributed: bool = True,
        combine: str = "by_coords",
        chunks: Optional[Mapping[str, int]] = None,
        parallel: bool = True,
        decode_times: bool = True,
        use_cftime: Optional[bool] = None,
        keep_variable_only: bool = False,
        drop_variables: Optional[Sequence[str]] = None,
        verbose: bool = False,
    ) -> xr.Dataset:
        """
        Open online ESGF files directly with xarray.

        This mirrors the notebook workflow: search ESGF File records, keep OPeNDAP
        URLs, deduplicate by time slice, then call ``xr.open_mfdataset``.
        """
        urls = self.online_urls_for_model(
            model=model,
            variable_id=variable_id,
            experiment_id=experiment_id,
            member_id=member_id,
            table_id=table_id,
            grid_label=grid_label,
            activity_id=activity_id,
            institution_id=institution_id,
            files_type=files_type,
            server=server,
            local_node=local_node,
            latest=latest,
            limit=limit,
            timeout=timeout,
            deduplicate=deduplicate,
            preferred_node=preferred_node,
            fallback_distributed=fallback_distributed,
            verbose=verbose,
        )
        kwargs: Dict[str, Any] = {
            "combine": combine,
            "chunks": dict(chunks or self.chunks),
            "parallel": parallel,
            "decode_times": decode_times,
        }
        if drop_variables is not None:
            kwargs["drop_variables"] = list(drop_variables)
        if use_cftime is not None:
            kwargs["use_cftime"] = use_cftime

        ds = xr.open_mfdataset(urls, **kwargs)
        if time_range is not None and "time" in ds.coords:
            ds = ds.sel(time=slice(*time_range))
        target_variable = variable_id or self._online_search_from_selection(
            model=model,
            variable_id=variable_id,
            experiment_id=experiment_id,
            member_id=member_id,
            table_id=table_id,
            grid_label=grid_label,
            activity_id=activity_id,
            institution_id=institution_id,
        )["variable_id"]
        if keep_variable_only:
            if target_variable not in ds:
                raise KeyError(f"Variable {target_variable!r} not found in online dataset")
            keep = [target_variable]
            keep.extend(coord for coord in CORE_COORDS if coord in ds.variables)
            ds = ds[sorted(set(keep))]
        ds.attrs["online_urls"] = "\n".join(urls)
        return ds

    def online_file_table_for_model(
        self,
        model: str,
        variable_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        member_id: Optional[str] = None,
        table_id: Optional[str] = None,
        grid_label: Optional[str] = None,
        activity_id: Optional[str] = None,
        institution_id: Optional[str] = None,
        files_type: str = "OPENDAP",
        server: str = "https://esgf-node.llnl.gov/esg-search/search",
        local_node: bool = True,
        latest: bool = True,
        limit: int = 10000,
        timeout: int = 60,
        fallback_distributed: bool = True,
        preferred_node: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Return a notebook-friendly table of online file URLs for one model.
        """
        urls = self.online_urls_for_model(
            model=model,
            variable_id=variable_id,
            experiment_id=experiment_id,
            member_id=member_id,
            table_id=table_id,
            grid_label=grid_label,
            activity_id=activity_id,
            institution_id=institution_id,
            files_type=files_type,
            server=server,
            local_node=local_node,
            latest=latest,
            limit=limit,
            timeout=timeout,
            fallback_distributed=fallback_distributed,
            preferred_node=preferred_node,
        )
        rows = []
        for url in urls:
            match = re.search(r"(\d{6}-\d{6})", url)
            rows.append(
                {
                    "timeslice": match.group(1) if match else None,
                    "node": self._node_from_url(url),
                    "url": url,
                }
            )
        return pd.DataFrame(rows)

    def _online_search_from_selection(
        self,
        model: str,
        variable_id: Optional[str],
        experiment_id: Optional[str],
        member_id: Optional[str],
        table_id: Optional[str],
        grid_label: Optional[str],
        activity_id: Optional[str],
        institution_id: Optional[str],
    ) -> Dict[str, Any]:
        rows = self._best_df[self._best_df["source_id"] == model]
        if rows.empty:
            raise KeyError(f"No selected row found for model={model!r}")
        if len(rows["variable_id"].unique()) > 1 and variable_id is None:
            available = sorted(rows["variable_id"].unique().tolist())
            raise ValueError(f"Model {model!r} has multiple variables; choose variable_id from {available}")

        row = rows.iloc[0]
        search = dict(self._constraints)
        search.update(
            {
                "source_id": model,
                "variable_id": variable_id or row["variable_id"],
                "member_id": member_id or row["member_id"],
                "grid_label": grid_label or row["grid_label"],
            }
        )
        if experiment_id is not None:
            search["experiment_id"] = experiment_id
        if table_id is not None:
            search["table_id"] = table_id
        if activity_id is not None:
            search["activity_id"] = activity_id
        if institution_id is not None:
            search["institution_id"] = institution_id
        return _normalize_constraints(search)

    @staticmethod
    def _node_from_url(url: str) -> str:
        match = re.match(r"^[a-z]+://([^/]+)", url)
        return match.group(1) if match else ""

    def _select_rows_for_requested_member_grid(
        self,
        model: str,
        experiment_id: Optional[str],
        variables: Optional[Iterable[str]],
        member_id: Optional[str],
        grid_label: Optional[str],
    ) -> pd.DataFrame:
        rows = self._df[self._df["source_id"] == model]
        if rows.empty:
            raise KeyError(f"Model {model!r} not found. Available models: {self.models_all()}")

        rows = self._select_experiment(rows, experiment_id, model=model)
        rows = self._select_variables(rows, variables, model=model)
        rows = self._select_member_grid(rows, member_id=member_id, grid_label=grid_label, model=model)
        return self._select_best_rows(rows)

    @staticmethod
    def _select_member_grid(
        rows: pd.DataFrame,
        member_id: Optional[str],
        grid_label: Optional[str],
        model: Optional[str] = None,
    ) -> pd.DataFrame:
        selected = rows
        if member_id is not None:
            selected = selected[selected["member_id"] == member_id]
        if grid_label is not None:
            selected = selected[selected["grid_label"] == grid_label]
        if not selected.empty:
            return selected

        prefix = f"{model}: " if model else ""
        available = (
            rows.loc[:, ["experiment_id", "variable_id", "member_id", "grid_label"]]
            .drop_duplicates()
            .sort_values(["experiment_id", "variable_id", "member_id", "grid_label"])
        )
        raise ValueError(
            f"{prefix}requested member_id={member_id!r}, grid_label={grid_label!r} is not available. "
            f"Available combinations:\n{available.to_string(index=False)}"
        )

    @staticmethod
    def _select_experiment(
        rows: pd.DataFrame,
        experiment_id: Optional[str],
        model: Optional[str] = None,
    ) -> pd.DataFrame:
        if experiment_id is not None:
            selected = rows[rows["experiment_id"] == experiment_id]
            if selected.empty:
                prefix = f"{model}: " if model else ""
                available = sorted(rows["experiment_id"].unique().tolist())
                raise ValueError(f"{prefix}experiment_id {experiment_id!r} not available. Available: {available}")
            return selected

        experiments = sorted(rows["experiment_id"].unique().tolist())
        if len(experiments) > 1:
            prefix = f"{model}: " if model else ""
            raise ValueError(
                f"{prefix}multiple experiment_id values are selected: {experiments}. "
                "Pass experiment_id='historical' or experiment_id='piControl'."
            )
        return rows

    @staticmethod
    def _group_open_rows(rows: pd.DataFrame):
        experiments = sorted(rows["experiment_id"].unique().tolist())
        group_cols = ["source_id"] if len(experiments) == 1 else ["source_id", "experiment_id"]
        for key, group in rows.groupby(group_cols):
            yield key, group

    @staticmethod
    def _select_variables(
        rows: pd.DataFrame,
        variables: Optional[Iterable[str]],
        model: Optional[str] = None,
    ) -> pd.DataFrame:
        if variables is None:
            return rows

        requested = tuple(variables)
        selected = rows[rows["variable_id"].isin(requested)]
        missing = sorted(set(requested) - set(selected["variable_id"]))
        if missing:
            prefix = f"{model}: " if model else ""
            available = sorted(rows["variable_id"].unique().tolist())
            raise ValueError(f"{prefix}variables not available: {missing}. Available: {available}")
        return selected

    def _open_rows(
        self,
        rows: pd.DataFrame,
        time_range: Optional[Tuple[str, str]],
        engine: Optional[str],
        decode_times: bool,
        use_cftime: Optional[bool],
        drop_conflicts: bool,
        join: str,
        keep_coords: Sequence[str],
        parallel: bool,
    ) -> xr.Dataset:
        datasets: List[xr.Dataset] = []
        for _, row in rows.iterrows():
            variable = str(row["variable_id"])
            kwargs: Dict[str, Any] = {
                "combine": "by_coords",
                "parallel": parallel,
                "chunks": dict(self.chunks),
                "engine": engine,
                "decode_times": decode_times,
                "preprocess": lambda ds, variable=variable: self._preprocess_dataset(ds, variable, keep_coords),
            }
            if use_cftime is not None:
                kwargs["use_cftime"] = use_cftime

            ds = xr.open_mfdataset(self._paths_for_choice(row), **kwargs)
            if time_range is not None and "time" in ds.coords:
                ds = ds.sel(time=slice(*time_range))
            datasets.append(ds)

        if not datasets:
            raise ValueError("No datasets selected to open")
        if len(datasets) == 1:
            return datasets[0]
        return xr.merge(
            datasets,
            compat="override" if drop_conflicts else "no_conflicts",
            join=join,
            combine_attrs="drop_conflicts",
        )

    @staticmethod
    def _selection_attrs(model: str, rows: pd.DataFrame) -> Dict[str, Any]:
        return {
            "model": model,
            "selection_experiment_ids": ",".join(sorted(rows["experiment_id"].unique())),
            "selection_variables": ",".join(sorted(rows["variable_id"].unique())),
            "selection_catalogs": ",".join(sorted(rows["catalog_label"].unique())),
            "selection_member_ids": ",".join(sorted(rows["member_id"].unique())),
            "selection_grid_labels": ",".join(sorted(rows["grid_label"].unique())),
            "selection_newest_version": int(rows["version_num"].max()),
        }

    @staticmethod
    def esgf_file_urls(
        server: str = "https://esgf-node.llnl.gov/esg-search/search",
        files_type: str = "OPENDAP",
        local_node: bool = True,
        project: str = "CMIP6",
        latest: bool = True,
        limit: int = 10000,
        timeout: int = 60,
        verbose: bool = False,
        **search: Any,
    ) -> List[str]:
        """
        Search ESGF File records and return URLs of the requested service type.

        ``files_type`` is usually ``"OPENDAP"`` for direct xarray access or
        ``"HTTPServer"`` for downloadable files.
        """
        files_type = files_type.upper()
        query_base = _normalize_constraints(search)
        query_base["project"] = project
        query_sets = _expanded_constraint_sets(
            query_base,
            (
                "activity_id",
                "experiment_id",
                "variable_id",
                "member_id",
                "table_id",
                "source_id",
                "grid_label",
                "institution_id",
            ),
        )

        session = requests.Session()
        urls: List[str] = []
        seen: set[str] = set()
        for query in query_sets:
            offset = 0
            num_found = 1
            while offset < num_found:
                params: Dict[str, Any] = {
                    "project": query.get("project", project),
                    "type": "File",
                    "latest": str(latest).lower(),
                    "format": "application/solr+json",
                    "limit": str(limit),
                    "offset": str(offset),
                }
                if local_node:
                    params["distrib"] = "false"
                for key, value in query.items():
                    if key == "project" or value is None:
                        continue
                    params[key] = value

                response = session.get(server, params=params, timeout=timeout)
                response.raise_for_status()
                payload = response.json()["response"]
                num_found = int(payload.get("numFound", 0))
                docs = payload.get("docs", [])
                if verbose:
                    print(response.url)
                    print(f"numFound={num_found}, docs={len(docs)}")
                if not docs:
                    break
                offset += len(docs)

                for doc in docs:
                    for entry in doc.get("url", []):
                        parts = entry.split("|")
                        if len(parts) < 3:
                            continue
                        if parts[-1].upper() != files_type:
                            continue
                        url = parts[0].split(".html")[0]
                        if url not in seen:
                            seen.add(url)
                            urls.append(url)
        return sorted(urls)

    @staticmethod
    def dedup_urls_by_timeslice(urls: Sequence[str], preferred_node: Optional[str] = None) -> List[str]:
        """
        Deduplicate CMIP-style file URLs by YYYYMM-YYYYMM time slice.

        If multiple nodes provide the same time slice, ``preferred_node`` wins.
        """
        kept: Dict[str, Tuple[int, str]] = {}
        for url in urls:
            match = re.search(r"(\d{6}-\d{6})", url)
            key = match.group(1) if match else url
            score = 0 if preferred_node and preferred_node in url else 1
            if key not in kept or score < kept[key][0]:
                kept[key] = (score, url)
        return sorted(item[1] for item in kept.values())

    @staticmethod
    def _latest_local_rows(df: pd.DataFrame) -> pd.DataFrame:
        _require_columns(
            df,
            ["source_id", "experiment_id", "member_id", "table_id", "variable_id", "grid_label", "version"],
            context="local catalog rows",
        )
        out = df.copy()
        out["version_num"] = out["version"].apply(_version_number)
        group_cols = ["source_id", "experiment_id", "member_id", "table_id", "variable_id", "grid_label"]
        return out.sort_values(group_cols + ["version_num"]).drop_duplicates(group_cols, keep="last")

    @staticmethod
    def _keys_from_rows(df: pd.DataFrame, include_version: bool) -> set[str]:
        columns = ["source_id", "experiment_id", "member_id", "table_id", "variable_id", "grid_label"]
        _require_columns(df, columns, context="catalog rows")
        if include_version:
            _require_columns(df, ["version"], context="catalog rows")
            columns = [*columns, "version"]

        keys = set()
        for _, row in df.iterrows():
            parts = [str(row[column]) for column in columns]
            if include_version:
                label = _version_label(parts[-1])
                if label is None:
                    continue
                parts[-1] = label
            keys.add(".".join(parts))
        return keys

    @staticmethod
    def _base_versions_from_rows(df: pd.DataFrame) -> Dict[Tuple[str, ...], set[str]]:
        _require_columns(
            df,
            ["source_id", "experiment_id", "member_id", "table_id", "variable_id", "grid_label", "version"],
            context="catalog rows",
        )
        out: Dict[Tuple[str, ...], set[str]] = {}
        for _, row in df.iterrows():
            base = tuple(
                str(row[column])
                for column in ["source_id", "experiment_id", "member_id", "table_id", "variable_id", "grid_label"]
            )
            label = _version_label(row["version"])
            if label is not None:
                out.setdefault(base, set()).add(label)
        return out

    @staticmethod
    def _base_versions_from_docs(docs: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, ...], set[str]]:
        out: Dict[Tuple[str, ...], set[str]] = {}
        for doc in docs:
            source = _first(doc.get("source_id"))
            experiment = _first(doc.get("experiment_id"))
            member = _first(doc.get("member_id"))
            table = _first(doc.get("table_id"))
            variable = _first(doc.get("variable_id"))
            grid = _first(doc.get("grid_label"))
            version = _first(doc.get("version"))

            if not all([source, experiment, member, table, variable, grid]):
                parsed = Ztake._parse_instance_id(_first(doc.get("instance_id")))
                if parsed is not None:
                    source, experiment, member, table, variable, grid, parsed_version = parsed
                    version = version or parsed_version

            if not all([source, experiment, member, table, variable, grid]) or version is None:
                continue

            label = _version_label(version)
            if label is None:
                continue
            base = tuple(str(part) for part in (source, experiment, member, table, variable, grid))
            out.setdefault(base, set()).add(label)
        return out

    @staticmethod
    def _parse_instance_id(instance_id: Any) -> Optional[Tuple[str, str, str, str, str, str, str]]:
        if not isinstance(instance_id, str):
            return None
        parts = instance_id.split(".")
        if len(parts) < 10:
            return None
        return parts[3], parts[4], parts[5], parts[6], parts[7], parts[8], parts[9]

    @staticmethod
    def _version_mismatch(
        local: Mapping[Tuple[str, ...], set[str]],
        online: Mapping[Tuple[str, ...], set[str]],
    ) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for base in sorted(set(local) & set(online)):
            local_versions = local[base]
            online_versions = online[base]
            if local_versions == online_versions:
                continue

            local_max = max((_version_number(v) for v in local_versions), default=0)
            online_max = max((_version_number(v) for v in online_versions), default=0)
            if local_max < online_max:
                status = "local older"
            elif local_max > online_max:
                status = "local newer"
            else:
                status = "different sets"

            out[".".join(base)] = {
                "local_versions": sorted(local_versions),
                "online_versions": sorted(online_versions),
                "local_max": f"v{local_max}" if local_versions else None,
                "online_max": f"v{online_max}" if online_versions else None,
                "status": status,
            }
        return out

    @staticmethod
    def _instance_ids_by_key(docs: Sequence[Mapping[str, Any]]) -> Dict[str, set[str]]:
        out: Dict[str, set[str]] = {}
        for doc in docs:
            base_versions = Ztake._base_versions_from_docs([doc])
            instance_id = _first(doc.get("instance_id"))
            if not instance_id:
                continue
            for base, versions in base_versions.items():
                for version in versions:
                    out.setdefault(".".join((*base, version)), set()).add(str(instance_id))
        return out

    @staticmethod
    def _esgf_query(
        constraints: Mapping[str, Any],
        latest: bool,
        limit: int,
        nodes: Optional[Sequence[str]],
        timeout: int,
    ) -> Tuple[Optional[str], List[Dict[str, Any]], Optional[str]]:
        esgf_filter_keys = (
            "experiment_id",
            "variable_id",
            "member_id",
            "table_id",
            "source_id",
            "grid_label",
            "institution_id",
            "activity_id",
        )
        query_constraints = _expanded_constraint_sets(constraints, esgf_filter_keys)

        headers = {"Accept": "application/solr+json", "User-Agent": "zpackage-esgf-compare"}
        last_error: Optional[str] = None
        first_successful_node: Optional[str] = None
        seen: set[str] = set()
        for node in nodes or ESGF_NODES_DEFAULT:
            node_docs: List[Dict[str, Any]] = []
            for query in query_constraints:
                params: Dict[str, Any] = {
                    "project": query.get("project", "CMIP6"),
                    "type": "Dataset",
                    "latest": str(latest).lower(),
                    "format": "application/solr+json",
                    "limit": str(limit),
                    "offset": "0",
                }
                for key in esgf_filter_keys:
                    value = query.get(key)
                    if value is not None:
                        params[key] = value

                try:
                    response = requests.get(node, params=params, headers=headers, timeout=timeout)
                    if response.status_code != 200:
                        last_error = f"HTTP {response.status_code} from {node}"
                        continue

                    first_successful_node = first_successful_node or node
                    data = response.json()
                    docs = data.get("response", {}).get("docs", [])
                    for doc in docs:
                        identity = str(doc.get("instance_id") or doc.get("id") or repr(sorted(doc.items())))
                        if identity in seen:
                            continue
                        seen.add(identity)
                        node_docs.append(dict(doc))
                except (requests.RequestException, ValueError) as exc:
                    last_error = f"{type(exc).__name__}: {exc}"

            if node_docs:
                return node, node_docs, None

        return first_successful_node, [], last_error

    def compare_with_esgf(
        self,
        mode: str = "latest",
        limit: int = 10000,
        nodes: Optional[Sequence[str]] = None,
        extra_constraints: Optional[Mapping[str, Any]] = None,
        return_ids: bool = True,
        request_ids: bool = False,
        filename: str = "only_online_instance_ids.txt",
        verbose: bool = False,
        timeout: int = 30,
    ) -> Dict[str, Any]:
        """
        Compare local intake rows against ESGF Solr search results.

        ``mode`` may be ``latest``, ``ignore_version``, or ``all_versions``.
        Empty online results are returned as an empty comparison instead of being
        treated as a transport failure.
        """
        constraints = dict(self._constraints)
        if extra_constraints:
            constraints.update(_normalize_constraints(extra_constraints))

        _, local_rows = self._search_catalogs(self._catalogs, constraints)
        if local_rows.empty:
            raise ValueError(f"No local catalog rows matched constraints: {constraints}")

        if mode == "latest":
            local_keys = self._keys_from_rows(self._latest_local_rows(local_rows), include_version=True)
            latest = True
        elif mode == "ignore_version":
            local_keys = self._keys_from_rows(local_rows, include_version=False)
            latest = True
        elif mode == "all_versions":
            local_keys = self._keys_from_rows(local_rows, include_version=True)
            latest = False
        else:
            raise ValueError("mode must be one of: 'latest', 'ignore_version', 'all_versions'")

        node, docs, error = self._esgf_query(
            constraints=constraints,
            latest=latest,
            limit=limit,
            nodes=nodes,
            timeout=timeout,
        )
        online_base_versions = self._base_versions_from_docs(docs)
        if mode == "ignore_version":
            online_keys = {".".join(base) for base in online_base_versions}
        else:
            online_keys = {".".join((*base, version)) for base, versions in online_base_versions.items() for version in versions}

        local_base_versions = self._base_versions_from_rows(local_rows)
        only_local = sorted(local_keys - online_keys)
        only_online = sorted(online_keys - local_keys)
        common = sorted(local_keys & online_keys)

        result: Dict[str, Any] = {
            "node": node,
            "mode": mode,
            "common": common,
            "only_local": only_local,
            "only_online": only_online,
            "local_count": len(local_keys),
            "online_count": len(online_keys),
            "version_mismatch": self._version_mismatch(local_base_versions, online_base_versions),
            "local_models": sorted(local_rows["source_id"].unique().tolist()) if "source_id" in local_rows else [],
            "online_models": sorted({_first(doc.get("source_id")) for doc in docs if doc.get("source_id")}),
            "error": error,
        }

        if return_ids:
            instance_ids_by_key = self._instance_ids_by_key(docs)
            instance_ids = sorted({iid for key in only_online for iid in instance_ids_by_key.get(key, set())})
            result["only_online_instance_ids"] = instance_ids
            if request_ids and instance_ids:
                self.save_ids_to_file(instance_ids, filename=filename)

        if verbose:
            print(f"Node: {result['node']}")
            print(f"Local datasets: {result['local_count']}")
            print(f"Online datasets: {result['online_count']}")
            print(f"Only online datasets: {len(result['only_online'])}")
            print(f"Version mismatches: {len(result['version_mismatch'])}")
            if error:
                print(f"Last ESGF query warning: {error}")

        return result

    @staticmethod
    def save_ids_to_file(ids: Sequence[str], filename: str, prefix: str = "instance_id") -> str:
        if prefix != "instance_id":
            raise ValueError("Only instance_id output is supported")

        path = Path(filename).expanduser().resolve()
        with path.open("w", encoding="utf-8") as handle:
            for item in ids:
                handle.write(f"{prefix}={item}\n")
        return str(path)
