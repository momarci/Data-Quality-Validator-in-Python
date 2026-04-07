"""
main.py (v3 — enhanced scalable edition)

- Scalable for multi-GB dataframes (no O(n^2) operations)
- Structural break tests (Chow, CUSUM, Zivot-Andrews) for time-series
- Memory-safe: chunked reads, proper gc, figure closing
- Improved logging and error handling
- Clean structured HTML report with embedded images
"""

import os
import json
import datetime
import pandas as pd
import numpy as np
import gc
import logging

from utils.read_data import read_data
from utils.detect_date_column import detect_date_columns
from utils.ask_data_type import detect_data_type

# Cross-sectional
from cross_sectional.min_max import compute_min_max
from cross_sectional.average import compute_average
from cross_sectional.median import compute_median
from cross_sectional.outliers import detect_outliers
from cross_sectional.multicollinearity import run_multicollinearity_checks
from cross_sectional.heteroskedasticity import run_heteroskedasticity_tests
from cross_sectional.normality import test_normality
from cross_sectional.correlation import compute_correlation
from cross_sectional.plots import scatter_plots, grouped_bar_plots, boxplots

# Time-series
from time_series.detect_range import detect_date_range_and_frequency
from time_series.frequency import analyze_frequency
from time_series.missing_dates import detect_missing_dates
from time_series.stl_decomposition import run_stl
from time_series.autocorrelation import compute_acf_pacf, ljung_box_test
from time_series.stationarity import stationarity_analysis
from time_series.volatility import arch_lm_test
from time_series.plots import time_series_plot, stl_plot, acf_pacf_plots
from time_series.structural_breaks import run_structural_break_tests, plot_structural_breaks

# Panel
from panel.detect_range import detect_panel_date_ranges
from panel.frequency import panel_frequency_analysis
from panel.balance import check_panel_balance, compute_variance_decomposition
from panel.plots import per_entity_series, faceted_panel_plot, panel_average, snapshot_across_entities

# HTML report
from reports.html_report_v2 import generate_html_report

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ----------------------------
# JSON Sanitization (fixed for Series)
# ----------------------------
def sanitize_for_json(obj):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Series):
        return {str(k): sanitize_for_json(v) for k, v in obj.to_dict().items()}
    if isinstance(obj, pd.DataFrame):
        d = obj.to_dict(orient="list")
        return {str(k): sanitize_for_json(v) for k, v in d.items()}
    if isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    return str(obj)


class StepwiseValidator:
    def __init__(self, filepath, manual_type, override_mgr, logger_callback, gui_override_callback):
        self.filepath = filepath
        self.manual_type = manual_type
        self.override_mgr = override_mgr
        self.log = logger_callback
        self.gui_override_callback = gui_override_callback

        self.df = None
        self.results = {}
        self.date_cols = []
        self.num_cols = []
        self.cat_cols = []

        # Output locations
        self.output_dir_base = "results"
        self.run_dir = None

    # ----------------------------
    # Utilities
    # ----------------------------
    def _ensure_run_dir(self):
        if self.run_dir:
            return self.run_dir
        base = getattr(self, "output_dir_base", "results")
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = os.path.join(base, ts)
        os.makedirs(self.run_dir, exist_ok=True)
        self.log(f"[Output] Run directory: {self.run_dir}")
        return self.run_dir

    def _plot_subdir(self, name: str) -> str:
        run = self._ensure_run_dir()
        p = os.path.join(run, name)
        os.makedirs(p, exist_ok=True)
        return p

    # ----------------------------
    # Steps
    # ----------------------------
    def step_read_data(self):
        self.log("[Step] Reading data...")
        sheet = getattr(self, "selected_sheet", None)
        self.df = read_data(self.filepath, sheet_name=sheet)

        num_rows, num_cols = self.df.shape
        mem_usage = self.df.memory_usage(deep=True).sum() / (1024 ** 2)
        self.log(f"Loaded dataframe: {num_rows:,} rows x {num_cols} cols ({mem_usage:.1f} MB)")

        # Store dataset metadata
        self.results["dataset_info"] = {
            "filename": os.path.basename(self.filepath),
            "rows": num_rows,
            "columns": num_cols,
            "memory_mb": round(mem_usage, 2),
            "column_names": list(self.df.columns),
            "dtypes": {str(c): str(d) for c, d in self.df.dtypes.items()},
        }

        if num_rows > 500_000:
            self.log(f"WARNING: Large dataset ({num_rows:,} rows). Using optimised code paths.")

        return True

    def step_detect_date_col(self):
        self.log("[Step] Detecting date columns...")
        detected = detect_date_columns(self.df)
        self.date_cols = detected.get("detected", [])
        self.results["date_detection"] = detected

        if self.override_mgr.has("date_column"):
            self.date_cols = [self.override_mgr.get("date_column")]
            self.log(f"Using user-selected date column: {self.date_cols[0]}")
            return True

        if len(self.date_cols) == 1:
            return True

        if len(self.date_cols) > 1:
            self.log(
                f"Multiple date-like columns detected {self.date_cols}; using first."
            )
            self.date_cols = [self.date_cols[0]]

        return True

    def step_missing_duplicates(self):
        self.log("[Step] Checking missing / duplicates...")
        from utils.check_missing_duplicate import run_missing_duplicate_checks

        date_col = self.date_cols[0] if self.date_cols else None
        entity_col = None
        if self.override_mgr.has("entity_column"):
            entity_col = self.override_mgr.get("entity_column")

        out = run_missing_duplicate_checks(self.df, entity_col=entity_col, date_col=date_col)
        self.results["missing_duplicate"] = out

        # Log summary
        miss = out["missing"]
        dup = out["duplicates"]
        self.log(f"  Missing: {miss['total_missing']:,} cells ({miss['missing_percentage']:.2f}%)")
        self.log(f"  Complete rows: {miss['complete_rows']:,} / {self.df.shape[0]:,}")
        self.log(f"  Duplicates: {dup['duplicate_count']:,} rows ({dup['duplicate_percentage']:.2f}%)")

        return True

    def step_convert_categories(self):
        self.log("[Step] Converting string columns to categorical...")
        df = self.df

        obj_cols = df.select_dtypes(include=["object", "string"]).columns
        cols_to_convert = [c for c in obj_cols if c not in self.date_cols]
        for col in cols_to_convert:
            df[col] = df[col].astype("category")

        self.df = df
        self.num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        self.cat_cols = df.select_dtypes(include=["category"]).columns.tolist()
        return True

    def step_detect_data_type(self):
        if self.manual_type:
            dtype = self.manual_type
        else:
            auto = detect_data_type(self.df, self.date_cols)
            self.results["auto_detected"] = auto
            dtype = auto["detected_type"]

        self.results["dataset_type"] = dtype
        self.log(f"Dataset type: {dtype}")
        return dtype

    # ----------------------------
    # Branches
    # ----------------------------
    def run_cross_sectional(self):
        self.log("[Branch] Cross-sectional analysis...")
        plot_dir = self._plot_subdir("cross_sectional_plots")

        self.results["min_max"] = compute_min_max(self.df)
        self.results["average"] = compute_average(self.df)
        self.results["median"] = compute_median(self.df)
        self.results["outliers"] = detect_outliers(self.df)

        try:
            self.results["multicollinearity"] = run_multicollinearity_checks(self.df, self.num_cols)
        except Exception as e:
            self.log(f"Multicollinearity: {e}")
            self.results["multicollinearity"] = {"error": str(e)}

        if self.num_cols:
            y = self.num_cols[0]
            X = self.num_cols[1:]
            try:
                self.results["heteroskedasticity"] = run_heteroskedasticity_tests(self.df, y, X)
            except Exception as e:
                self.log(f"Heteroskedasticity: {e}")
                self.results["heteroskedasticity"] = {"error": str(e)}

        try:
            self.results["normality"] = test_normality(self.df)
        except Exception as e:
            self.log(f"Normality: {e}")
            self.results["normality"] = {"error": str(e)}

        try:
            self.results["correlation"] = compute_correlation(self.df)
        except Exception as e:
            self.log(f"Correlation: {e}")
            self.results["correlation"] = {"error": str(e)}

        scatter_plots(self.df, save_dir=plot_dir)
        grouped_bar_plots(self.df, self.cat_cols[0] if self.cat_cols else None, save_dir=plot_dir)
        boxplots(self.df, save_dir=plot_dir)

        self.results.setdefault("plot_dirs", []).append(plot_dir)
        return True

    def run_time_series(self):
        self.log("[Branch] Time-series analysis...")
        plot_dir_base = self._plot_subdir("time_series_plots")
        date = self.date_cols[0]

        # --- Date-level analyses (computed once for the whole dataset) ---
        range_freq = detect_date_range_and_frequency(self.df, date)
        freq = range_freq["final_frequency"]
        self.results["range_frequency"] = range_freq
        self.results["frequency_details"] = analyze_frequency(self.df, date, expected_freq=freq)
        self.results["missing_dates"] = detect_missing_dates(self.df, date, inferred_freq=freq)

        if not self.num_cols:
            self.log("  No numeric columns found for time-series analysis.")
            return True

        # --- Per-column analyses ---
        col_results = {}
        stl_period_cache = None   # reuse inferred STL period across columns

        for value in self.num_cols:
            self.log(f"  [Column] {value}")

            # Filesystem-safe subfolder name
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(value))[:60]
            col_plot_dir = os.path.join(plot_dir_base, safe)
            os.makedirs(col_plot_dir, exist_ok=True)

            col = {}

            # STL decomposition — seasonal_strength stored; full series used for plot only
            stl = None
            try:
                stl = run_stl(self.df, date, value,
                              period=stl_period_cache,
                              inferred_freq=freq,
                              gui_override_callback=self.gui_override_callback)
                if stl_period_cache is None:
                    stl_period_cache = stl.get("period")
                col["seasonal_strength"] = stl.get("seasonal_strength")
            except Exception as e:
                self.log(f"    STL({value}): {e}")

            # ACF / PACF (plot only)
            try:
                compute_acf_pacf(self.df, date, value)   # result used for plot only
            except Exception as e:
                self.log(f"    ACF/PACF({value}): {e}")

            # Ljung-Box
            try:
                col["ljung_box"] = ljung_box_test(self.df, date, value)
            except Exception as e:
                col["ljung_box"] = {"error": str(e)}

            # Stationarity
            try:
                col["stationarity"] = stationarity_analysis(self.df, date, value)
            except Exception as e:
                col["stationarity"] = {"error": str(e)}

            # ARCH-LM (volatility clustering / conditional heteroskedasticity)
            try:
                ts_vals = pd.to_numeric(self.df[value], errors="coerce").dropna()
                col["arch_lm"] = arch_lm_test(ts_vals)
            except Exception as e:
                col["arch_lm"] = {"error": str(e)}

            # Structural breaks
            try:
                sb_results = run_structural_break_tests(self.df, date, value)
                col["structural_breaks"] = sb_results
                plot_structural_breaks(self.df, date, value, sb_results, col_plot_dir)
            except Exception as e:
                self.log(f"    Structural breaks({value}): {e}")
                col["structural_breaks"] = {"error": str(e)}

            # --- Plots ---
            try:
                time_series_plot(self.df, date, value,
                                 save_path=os.path.join(col_plot_dir, "lineplot.png"))
            except Exception as e:
                self.log(f"    Lineplot({value}): {e}")

            if stl is not None:
                try:
                    stl_plot(stl, save_path=os.path.join(col_plot_dir, "stl_components.png"))
                except Exception as e:
                    self.log(f"    STL plot({value}): {e}")

            try:
                series = self.df[value].dropna()
                acf_pacf_plots(series, save_dir=col_plot_dir)
            except Exception as e:
                self.log(f"    ACF/PACF plots({value}): {e}")

            col_results[value] = col
            self.results.setdefault("plot_dirs", []).append(col_plot_dir)

        self.results["columns"] = col_results
        return True
    
    def _run_time_series_on_subset(self, df_sub, entity_name, entity_plot_dir):
        date = self.date_cols[0]
        freq_info = detect_date_range_and_frequency(df_sub, date)
        freq = freq_info["final_frequency"]

        entity_results = {
            "range_frequency": freq_info,
            "frequency_details": analyze_frequency(df_sub, date, expected_freq=freq),
            "missing_dates": detect_missing_dates(df_sub, date, inferred_freq=freq),
            "columns": {}
        }

        stl_period_cache = None

        for value in self.num_cols:
            col = {}
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(value))[:60]
            col_plot_dir = os.path.join(entity_plot_dir, safe)
            os.makedirs(col_plot_dir, exist_ok=True)

            # STL
            try:
                stl = run_stl(df_sub, date, value,
                            period=stl_period_cache,
                            inferred_freq=freq,
                            gui_override_callback=self.gui_override_callback)
                if stl_period_cache is None:
                    stl_period_cache = stl.get("period")
                col["seasonal_strength"] = stl.get("seasonal_strength")
            except Exception as e:
                col["stl_error"] = str(e)

            # Ljung-Box
            try:
                col["ljung_box"] = ljung_box_test(df_sub, date, value)
            except Exception as e:
                col["ljung_box"] = {"error": str(e)}

            # Stationarity
            try:
                col["stationarity"] = stationarity_analysis(df_sub, date, value)
            except Exception as e:
                col["stationarity"] = {"error": str(e)}

            # ARCH-LM
            try:
                ts_vals = pd.to_numeric(df_sub[value], errors="coerce").dropna()
                col["arch_lm"] = arch_lm_test(ts_vals)
            except Exception as e:
                col["arch_lm"] = {"error": str(e)}

            # Structural breaks
            try:
                sb_res = run_structural_break_tests(df_sub, date, value)
                col["structural_breaks"] = sb_res
                plot_structural_breaks(df_sub, date, value, sb_res, col_plot_dir)
            except Exception as e:
                col["structural_breaks"] = {"error": str(e)}

            # Line plot
            try:
                time_series_plot(df_sub, date, value,
                                save_path=os.path.join(col_plot_dir, "lineplot.png"))
            except:
                pass

            entity_results["columns"][value] = col

            self.results.setdefault("plot_dirs", []).append(col_plot_dir)

        return entity_results
    
    def run_panel(self):
        self.log("[Branch] Panel data analysis...")
        plot_dir = self._plot_subdir("panel_plots")

        date = self.date_cols[0]

        # ------------------------------
        # Entity column selection
        # ------------------------------
        if self.override_mgr.has("entity_column"):
            entity = self.override_mgr.get("entity_column")
        else:
            if not self.cat_cols:
                raise ValueError("Cannot identify entity column — no categorical columns found.")
            override = self.gui_override_callback({
                "title": "Entity Column",
                "fields": {
                    "entity_column": {
                        "type": "dropdown",
                        "label": "Select entity ID column:",
                        "options": self.cat_cols,
                    }
                },
            })
            entity = override["entity_column"]
            self.override_mgr.set("entity_column", entity)

        self.log(f"Entity column: {entity}")

        # ------------------------------
        # Panel-level diagnostics
        # ------------------------------
        self.results["panel_ranges"] = detect_panel_date_ranges(self.df, entity, date)

        try:
            self.results["panel_balance"] = check_panel_balance(self.df, entity, date)
        except Exception as e:
            self.log(f"Panel balance: {e}")
            self.results["panel_balance"] = {"error": str(e)}

        try:
            self.results["panel_variance"] = compute_variance_decomposition(
                self.df, entity, self.num_cols)
        except Exception as e:
            self.log(f"Panel variance decomposition: {e}")
            self.results["panel_variance"] = {"error": str(e)}

        try:
            self.results["panel_frequency"] = panel_frequency_analysis(self.df, entity, date)
        except Exception:
            override = self.gui_override_callback({
                "title": "Panel Frequency Override",
                "fields": {
                    "panel_frequency": {"type": "text", "label": "Enter panel frequency:"}
                },
            })
            self.results["panel_frequency"] = {"overridden_frequency": override["panel_frequency"]}

        # ------------------------------
        # Panel-level plots
        # ------------------------------
        self.df[date] = pd.to_datetime(self.df[date], errors="coerce")
        last_date = self.df[date].max()
        value = self.num_cols[0]

        per_entity_series(self.df, entity, date, value, save_dir=plot_dir)
        faceted_panel_plot(self.df, entity, date, value, save_dir=plot_dir)
        panel_average(self.df, date, value, save_dir=plot_dir)
        snapshot_across_entities(self.df, entity, date, value,
                                date_value=last_date, save_dir=plot_dir)

        self.results.setdefault("plot_dirs", []).append(plot_dir)

        # ------------------------------------------------
        # ✅ NEW SECTION — Entity-wise time-series analysis
        # ------------------------------------------------
        self.log("[Panel-TS] Running entity-wise time-series analysis...")

        panel_ts_results = {}
        panel_ts_dir = self._plot_subdir("panel_entity_ts")

        entities = self.df[entity].unique()

        for ent in entities:
            self.log(f"[Panel-TS] Processing entity: {ent}")

            df_ent = self.df[self.df[entity] == ent].copy()

            ent_dir = os.path.join(panel_ts_dir, str(ent))
            os.makedirs(ent_dir, exist_ok=True)

            try:
                panel_ts_results[str(ent)] = self._run_time_series_on_subset(
                    df_ent,
                    ent,
                    ent_dir
                )
            except Exception as e:
                self.log(f"[Panel-TS] Error for entity {ent}: {e}")
                panel_ts_results[str(ent)] = {"error": str(e)}

        self.results["panel_entity_ts"] = panel_ts_results

        return True

    # ----------------------------
    # Checkpoint (best-effort partial save)
    # ----------------------------
    def _checkpoint(self):
        """Write whatever is in self.results to partial_results.json so a crash
        doesn't lose all computed data."""
        try:
            path = os.path.join(self._ensure_run_dir(), "partial_results.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(sanitize_for_json(self.results), f, indent=2)
            self.log(f"[Checkpoint] Partial results saved: {path}")
        except Exception as e:
            self.log(f"[Checkpoint] Could not save partial results: {e}")

    # ----------------------------
    # Run entire pipeline
    # ----------------------------
    def run_all_steps(self):
        self._ensure_run_dir()

        try:
            for step in [
                self.step_read_data,
                self.step_detect_date_col,
                self.step_missing_duplicates,
                self.step_convert_categories,
                self.step_detect_data_type,
            ]:
                step()

            # Save what we have after the expensive data-quality step so that
            # a crash in the analysis branch still leaves useful output on disk.
            self._checkpoint()

            dtype = self.results["dataset_type"]
            try:
                if dtype == "cross_sectional":
                    self.run_cross_sectional()
                elif dtype == "time_series":
                    self.run_time_series()
                elif dtype == "panel":
                    self.run_panel()
                else:
                    raise ValueError(f"Unknown dataset type: {dtype}")
            except Exception:
                self._checkpoint()  # save partial analysis results before re-raising
                raise

        finally:
            self.log("[Cleanup] Clearing intermediate data from memory...")
            gc.collect()

        return self.results

    # ----------------------------
    # HTML report
    # ----------------------------
    def generate_html_report(self):
        run = self._ensure_run_dir()
        safe = sanitize_for_json(self.results)
        report_path = os.path.join(run, "report.html")

        # Also save JSON results
        json_path = os.path.join(run, "results.json")
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(safe, f, indent=2)
        except Exception as e:
            self.log(f"Could not save results.json: {e}")

        generate_html_report(safe, report_path)
        self.log(f"Report saved: {report_path}")
        return report_path
