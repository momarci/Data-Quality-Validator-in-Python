"""
reports/html_report_v2.py  (v4 — tight, clean report)

Design:
  - Custom renderer per result key — no generic key-value dumps
  - Raw series/arrays (plot back-data) never shown in tables
  - Plots embedded inline within their relevant section
  - Clean, readable column names in every table
  - Self-contained single-file HTML (base64 images)
"""
from __future__ import annotations

import os
import re
import base64
import datetime
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

_SAFE = re.compile(r"[^a-zA-Z0-9_\-]+")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _esc(s: Any) -> str:
    s = "" if s is None else str(s)
    return (s.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;").replace('"', "&quot;"))


def _safe_id(name: str) -> str:
    return _SAFE.sub("_", (name or "section").strip()).strip("_") or "section"


def _img_b64(img_path: str) -> str:
    try:
        with open(img_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
        ext = Path(img_path).suffix.lower().lstrip(".")
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "svg": "image/svg+xml"}.get(ext, "image/png")
        return f"data:{mime};base64,{data}"
    except Exception:
        return ""


def _fmt(v: Any, decimals: int = 4) -> str:
    """Format a scalar value for display."""
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "Yes" if v else "No"
    if isinstance(v, float):
        return f"{v:,.2f}" if abs(v) >= 1000 else f"{v:.{decimals}f}"
    if isinstance(v, int):
        return f"{v:,}"
    return _esc(str(v))


def _pval_badge(pvalue: Any) -> str:
    """Coloured badge for a p-value."""
    try:
        p = float(pvalue)
        if p < 0.01:
            return f'<span class="badge bad">p = {p:.4f} ***</span>'
        if p < 0.05:
            return f'<span class="badge bad">p = {p:.4f} **</span>'
        if p < 0.10:
            return f'<span class="badge warn">p = {p:.4f} *</span>'
        return f'<span class="badge ok">p = {p:.4f}</span>'
    except (TypeError, ValueError):
        return _esc(str(pvalue))


def _bool_badge(val: Any, good_when_true: bool = True) -> str:
    is_true = bool(val)
    cls = "ok" if (is_true == good_when_true) else "bad"
    return f'<span class="badge {cls}">{"Yes" if is_true else "No"}</span>'


def _table(headers: List[str], rows: List[List[Any]], caption: str = "") -> str:
    """Render a styled HTML table.  Cell values that already contain HTML are
    passed through; plain strings are escaped."""
    out: List[str] = []
    if caption:
        out.append(f'<p class="tbl-caption">{_esc(caption)}</p>')
    out.append('<div class="tbl-wrap"><table>')
    out.append('<thead><tr>')
    for h in headers:
        out.append(f'<th>{_esc(h)}</th>')
    out.append('</tr></thead><tbody>')
    for row in rows:
        out.append('<tr>')
        for cell in row:
            s = str(cell) if cell is not None else "—"
            out.append(f'<td>{"" if cell is None else (s if "<" in s else _esc(s))}</td>')
        out.append('</tr>')
    out.append('</tbody></table></div>')
    return "".join(out)


def _kv(pairs: List[Tuple[str, Any]]) -> str:
    """Two-column label / value table."""
    out = ['<div class="tbl-wrap"><table class="kv">']
    for label, value in pairs:
        v = str(value) if value is not None else "—"
        out.append(f'<tr><th>{_esc(label)}</th><td>{"" if value is None else (v if "<" in v else _esc(v))}</td></tr>')
    out.append('</table></div>')
    return "".join(out)


def _section_html(title: str, anchor: str, body: str) -> str:
    return (f'<div class="section" id="{_esc(anchor)}">'
            f'<h2>{_esc(title)}</h2>{body}</div>')


def _h3(text: str) -> str:
    return f'<h3>{_esc(text)}</h3>'


# ---------------------------------------------------------------------------
# Plot embedding
# ---------------------------------------------------------------------------

def _embed_plots(plot_dirs: List[str], name_filter: str = "") -> str:
    """Return HTML for all PNGs in directories whose name contains name_filter."""
    parts: List[str] = []
    for folder in (plot_dirs or []):
        if not folder or not os.path.exists(folder):
            continue
        if name_filter and name_filter.lower() not in Path(folder).name.lower():
            continue
        pngs = sorted(p for p in os.listdir(folder)
                      if p.lower().endswith(".png") and "thumbs" not in p.lower())
        if not pngs:
            continue
        parts.append('<div class="plot-grid">')
        for png in pngs:
            b64 = _img_b64(os.path.join(folder, png))
            if b64:
                label = Path(png).stem.replace("_", " ").title()
                parts.append(
                    f'<figure class="plot-fig">'
                    f'<img src="{b64}" alt="{_esc(label)}" loading="lazy"/>'
                    f'<figcaption>{_esc(label)}</figcaption>'
                    f'</figure>'
                )
        parts.append('</div>')
    return "".join(parts)


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def _r_dataset_info(info: dict) -> str:
    col_names = info.get("column_names", [])
    pairs = [
        ("Filename",    _esc(info.get("filename", "—"))),
        ("Shape",       f"{_fmt(info.get('rows', '?'))} rows × {info.get('columns', '?')} columns"),
        ("Memory",      f"{info.get('memory_mb', '—')} MB"),
        ("Columns",     _esc(", ".join(str(c) for c in col_names)) if col_names else "—"),
    ]
    return _kv(pairs)


def _r_missing_duplicate(data: dict) -> str:
    out: List[str] = []
    miss = data.get("missing", {})
    dups = data.get("duplicates", {})

    if miss:
        out.append(_h3("Missing Values"))
        out.append(_kv([
            ("Total missing cells",
             f"{_fmt(miss.get('total_missing', 0))} "
             f"({miss.get('missing_percentage', 0):.2f}% of all cells)"),
            ("Complete rows",
             f"{_fmt(miss.get('complete_rows', 0))} "
             f"({miss.get('complete_row_percentage', 0):.1f}%)"),
        ]))

        per_col = miss.get("missing_per_column", {})
        pct_col = miss.get("missing_pct_per_column", {})
        if per_col:
            rows = [
                [_esc(col), _fmt(cnt), f"{pct_col.get(col, 0):.2f}%"]
                for col, cnt in sorted(per_col.items(), key=lambda x: -x[1])
            ]
            out.append(_table(["Column", "Missing Count", "Missing %"], rows,
                               "Columns with missing values"))

        bands = miss.get("row_completeness_bands", {})
        if bands:
            out.append(_table(
                ["Completeness Band", "Row Count"],
                [[_esc(b), _fmt(c)] for b, c in bands.items()],
                "Row completeness distribution"
            ))

    if dups:
        out.append(_h3("Duplicates"))
        rc = dups.get("rule_counts", {})
        out.append(_kv([
            ("Duplicate rows",
             f"{_fmt(dups.get('duplicate_count', 0))} "
             f"({dups.get('duplicate_percentage', 0):.2f}%)"),
            ("Exact full-row duplicates (Rule A)", _fmt(rc.get("rule_a", 0))),
            ("Same-date near-duplicates (Rule B)", _fmt(rc.get("rule_b", 0))),
            ("Same entity+date near-duplicates (Rule C)", _fmt(rc.get("rule_c", 0))),
        ]))

        groups = dups.get("duplicate_groups", [])
        if groups:
            # Collect key column names from first group
            key_cols = list(groups[0]["key_values"].keys()) if groups else []
            headers = ["Group", "Row Indices", "# Rows"] + [_esc(c) for c in key_cols]
            rows = []
            for i, g in enumerate(groups, 1):
                indices_str = ", ".join(str(x) for x in g["indices"])
                row = [str(i), indices_str, str(len(g["indices"]))]
                for col in key_cols:
                    row.append(_esc(g["key_values"].get(col, "—")))
                rows.append(row)
            total = dups.get("duplicate_count", 0)
            cap = f"Exact duplicate groups — Rule A"
            if dups.get("duplicates_truncated"):
                cap += f" (showing {len(groups)} of {total} affected rows)"
            out.append(_table(headers, rows, cap))

    return "".join(out)


def _r_descriptive_stats(min_max: Optional[dict],
                          avg: Optional[dict],
                          med: Optional[dict],
                          outliers: Optional[dict]) -> str:
    out: List[str] = []

    # Merge min/max, average, median into one table
    all_cols: List[str] = []
    for d in (min_max, avg, med):
        if d:
            all_cols += [c for c in d if c not in all_cols]

    if all_cols and (min_max or avg or med):
        headers = ["Column"]
        if min_max: headers += ["Min", "Max"]
        if avg:     headers += ["Mean", "Std Dev", "Skewness", "Kurtosis"]
        if med:     headers.append("Median")
        rows = []
        for col in all_cols:
            row: List[str] = [_esc(col)]
            if min_max:
                v = min_max.get(col, {})
                row += [_fmt(v.get("min") if isinstance(v, dict) else None),
                        _fmt(v.get("max") if isinstance(v, dict) else None)]
            if avg:
                v = avg.get(col)
                if isinstance(v, dict):
                    row += [
                        _fmt(v.get("mean")),
                        _fmt(v.get("std")),
                        _fmt(v.get("skewness"), 4),
                        _fmt(v.get("kurtosis"), 4),
                    ]
                else:
                    row += [_fmt(v), "—", "—", "—"]
            if med:
                v = med.get(col)
                row.append(_fmt(v.get("median") if isinstance(v, dict) else v))
            rows.append(row)
        out.append(_h3("Summary Statistics"))
        out.append(_table(headers, rows))

    if outliers:
        rows = []
        for col, v in outliers.items():
            if not isinstance(v, dict):
                continue
            cnt = int(v.get("count", 0))
            badge = (f'<span class="badge {"bad" if cnt > 0 else "ok"}">{_fmt(cnt)}</span>')
            rows.append([
                _esc(col),
                _esc(v.get("method", "IQR")),
                _fmt(v.get("lower_fence")),
                _fmt(v.get("upper_fence")),
                badge,
            ])
        if rows:
            out.append(_h3("Outlier Detection"))
            out.append(_table(
                ["Column", "Method", "Lower Fence", "Upper Fence", "Outlier Count"],
                rows,
                "Outlier counts only — individual values omitted"
            ))

    return "".join(out)


def _r_regression_diagnostics(mc: Optional[dict], hk: Optional[dict]) -> str:
    out: List[str] = []

    if mc:
        vif = mc.get("vif", {})
        cond = mc.get("condition_number")
        if vif:
            rows = []
            for col, v in sorted(vif.items(), key=lambda x: -(x[1] or 0)):
                if v is None:
                    badge = "—"
                elif v > 10:
                    badge = f'<span class="badge bad">High (&gt;10)</span>'
                elif v > 5:
                    badge = f'<span class="badge warn">Moderate (5–10)</span>'
                else:
                    badge = f'<span class="badge ok">Low (≤ 5)</span>'
                rows.append([_esc(col), _fmt(v, 2), badge])
            out.append(_h3("Variance Inflation Factors (VIF)"))
            out.append(_table(["Column", "VIF", "Severity"], rows))
        if cond is not None:
            if cond > 30:
                label = f'<span class="badge bad">Severe (&gt; 30)</span>'
            elif cond > 15:
                label = f'<span class="badge warn">Moderate (15–30)</span>'
            else:
                label = f'<span class="badge ok">Low (≤ 15)</span>'
            out.append(_kv([("Condition Number", f"{_fmt(cond, 2)} — {label}")]))

    if hk:
        rows = []
        for name, vals in [("Breusch-Pagan", hk.get("breusch_pagan", {})),
                            ("White Test",   hk.get("white_test", {}))]:
            if not vals or "error" in vals:
                continue
            lm_p = vals.get("lm_pvalue")
            f_p  = vals.get("f_pvalue")
            try:
                hetero = float(lm_p) < 0.05
                result = (f'<span class="badge {"bad" if hetero else "ok"}">'
                          f'{"Heteroskedastic" if hetero else "Homoskedastic"}</span>')
            except (TypeError, ValueError):
                result = "—"
            rows.append([
                _esc(name),
                _fmt(vals.get("lm"), 4),
                _pval_badge(lm_p),
                _fmt(vals.get("f_stat"), 4),
                _pval_badge(f_p),
                result,
            ])
        if rows:
            out.append(_h3("Heteroskedasticity Tests"))
            out.append(_table(
                ["Test", "LM Stat", "LM p-value", "F-Stat", "F p-value", "Result"],
                rows
            ))

    return "".join(out)


def _r_ts_coverage(rf: Optional[dict],
                   fd: Optional[dict],
                   mdf: Optional[dict]) -> str:
    out: List[str] = []

    if rf:
        out.append(_h3("Date Range & Frequency"))
        out.append(_kv([
            ("Start Date",        _esc(str(rf.get("start_date", "—")))),
            ("End Date",          _esc(str(rf.get("end_date", "—")))),
            ("Inferred Frequency", _esc(str(rf.get("inferred_frequency",
                                                    rf.get("final_frequency", "—"))))),
            ("Total Records",     _fmt(rf.get("total_records", rf.get("n_observations")))),
        ]))

    if fd:
        out.append(_h3("Frequency Diagnostics"))
        ev_match = fd.get("expected_vs_observed_match")
        out.append(_kv([
            ("Observations",          _fmt(fd.get("n_observations"))),
            ("Most Common Interval",  _esc(str(fd.get("most_common_delta", "—")))),
            ("Regular Series",        _bool_badge(fd.get("is_regular"), good_when_true=True)),
            ("Matches Expected Freq", _bool_badge(ev_match, good_when_true=True)
                                      if ev_match is not None else "—"),
        ]))

        # Suspicious gaps
        gaps = fd.get("suspicious_gaps")
        try:
            gap_list = (list(gaps.itertuples(index=False, name=None))
                        if hasattr(gaps, "itertuples") else
                        gaps if isinstance(gaps, list) else [])
            if gap_list:
                gap_rows = [[_esc(str(g[0])), _esc(str(g[1])), _esc(str(g[2]))]
                             for g in gap_list[:50]]
                out.append(_table(
                    ["Gap Start", "Gap End", "Duration"],
                    gap_rows,
                    f"Suspicious gaps (showing up to 50 of {len(gap_list)})"
                ))
        except Exception:
            pass

    if mdf:
        out.append(_h3("Missing Dates"))
        out.append(_kv([
            ("Start Date",      _esc(str(mdf.get("start", "—")))),
            ("End Date",        _esc(str(mdf.get("end", "—")))),
            ("Frequency",       _esc(str(mdf.get("frequency", "—")))),
            ("Expected Dates",  _fmt(mdf.get("expected_length"))),
            ("Actual Dates",    _fmt(mdf.get("actual_length"))),
            ("Missing Dates",   _fmt(mdf.get("missing_count", 0))),
        ]))

    return "".join(out)


def _r_stationarity(data: dict) -> str:
    tests = [
        ("ADF Test",             data.get("adf", {})),
        ("Phillips-Perron",      data.get("pp", {})),
        ("KPSS (Level)",         data.get("kpss_level", {})),
        ("KPSS (Trend)",         data.get("kpss_trend", {})),
    ]
    rows = []
    for name, t in tests:
        if not t or "error" in t:
            continue
        is_stat = t.get("stationary")
        conclusion = (f'<span class="badge {"ok" if is_stat else "bad"}">'
                      f'{"Stationary" if is_stat else "Non-stationary"}</span>')
        rows.append([
            _esc(name),
            _fmt(t.get("statistic"), 4),
            _pval_badge(t.get("pvalue")),
            _fmt(t.get("lags")),
            _fmt(t.get("n_obs")),
            conclusion,
        ])
    out = [_table(
        ["Test", "Statistic", "p-value", "Lags", "N Obs", "Conclusion"], rows
    )]

    # ADF critical values
    cv = (data.get("adf") or {}).get("critical_values", {})
    if cv:
        out.append(_table(
            ["Significance Level", "ADF Critical Value"],
            [[_esc(k), _fmt(v, 4)] for k, v in cv.items()],
            "ADF critical values"
        ))
    return "".join(out)


def _r_ljung_box(data: dict) -> str:
    pval = data.get("pvalue")
    try:
        is_ac = float(pval) < 0.05
    except (TypeError, ValueError):
        is_ac = False
    result = (f'<span class="badge {"bad" if is_ac else "ok"}">'
              f'{"Autocorrelation detected" if is_ac else "No significant autocorrelation"}</span>')
    rows = [[_fmt(data.get("lag")), _fmt(data.get("lb_stat"), 4), _pval_badge(pval), result]]
    return _table(["Lag", "LB Statistic", "p-value", "Conclusion"], rows)


def _r_ts_column(col_name: str, col_data: dict) -> str:
    """Render all per-column time-series results: stationarity, LB, structural breaks."""
    out: List[str] = []
    out.append(f'<h3>Column: {_esc(str(col_name))}</h3>')

    # Seasonal strength (from STL)
    ss = col_data.get("seasonal_strength")
    if ss is not None:
        try:
            ss_f = float(ss)
            if ss_f >= 0.64:
                ss_cls, ss_label = "bad",  "Strong"
            elif ss_f >= 0.36:
                ss_cls, ss_label = "warn", "Moderate"
            else:
                ss_cls, ss_label = "ok",   "Weak"
            out.append(
                f'<p class="subsec-lbl">Seasonal Strength (STL): '
                f'{ss_f:.4f} — <span class="badge {ss_cls}">{ss_label}</span></p>'
            )
        except (TypeError, ValueError):
            pass

    stat = col_data.get("stationarity")
    if stat and "error" not in stat:
        out.append('<p class="subsec-lbl">Stationarity Tests</p>')
        out.append(_r_stationarity(stat))
    elif stat and "error" in stat:
        out.append(f'<p class="muted">Stationarity: {_esc(stat["error"])}</p>')

    lb = col_data.get("ljung_box")
    if lb and "error" not in lb:
        out.append('<p class="subsec-lbl">Autocorrelation — Ljung-Box</p>')
        out.append(_r_ljung_box(lb))
    elif lb and "error" in lb:
        out.append(f'<p class="muted">Ljung-Box: {_esc(lb["error"])}</p>')

    arch = col_data.get("arch_lm")
    if arch and "error" not in arch:
        out.append('<p class="subsec-lbl">Volatility — ARCH-LM Test</p>')
        out.append(_r_arch_lm(arch))
    elif arch and "error" in arch:
        out.append(f'<p class="muted">ARCH-LM: {_esc(arch["error"])}</p>')

    sb = col_data.get("structural_breaks")
    if sb and "error" not in sb:
        out.append('<p class="subsec-lbl">Structural Break Tests</p>')
        out.append(_r_structural_breaks(sb))
    elif sb and "error" in sb:
        out.append(f'<p class="muted">Structural breaks: {_esc(sb["error"])}</p>')

    return "".join(out)


def _r_structural_breaks(data: dict) -> str:
    """Summary table per test — raw arrays (cusum_values, all_candidates) never shown."""
    out: List[str] = []
    rows = []

    chow = data.get("chow") or {}
    if "error" not in chow:
        bb = chow.get("best_break") or {}
        if bb and "error" not in bb:
            rows.append([
                "Chow Test",
                f"F = {_fmt(bb.get('f_statistic'), 3)}",
                _pval_badge(bb.get("p_value")),
                _fmt(bb.get("break_index")),
                _bool_badge(bb.get("significant", False), good_when_true=False),
            ])

    cusum = data.get("cusum") or {}
    if "error" not in cusum and cusum:
        bnd = cusum.get("boundary_5pct", 0.948)
        rows.append([
            "CUSUM Test",
            f"Max = {_fmt(cusum.get('max_cusum'), 4)} (boundary ±{bnd})",
            "—",
            _fmt(cusum.get("break_index")),
            _bool_badge(cusum.get("rejects_stability", False), good_when_true=False),
        ])

    za = data.get("zivot_andrews") or {}
    if "error" not in za and za:
        rows.append([
            "Zivot-Andrews",
            f"ZA stat = {_fmt(za.get('za_statistic'), 4)}",
            _pval_badge(za.get("p_value")),
            _fmt(za.get("break_index")),
            _bool_badge(za.get("stationary_with_break", False), good_when_true=False),
        ])

    sim = data.get("shift_in_means") or {}
    sim_bb = sim.get("best_break") or {}
    if "error" not in sim and sim_bb:
        rows.append([
            "Shift in Means",
            f"t = {_fmt(sim_bb.get('t_statistic'), 3)}, "
            f"Δμ = {_fmt(sim_bb.get('shift_magnitude'), 4)}",
            _pval_badge(sim_bb.get("p_value")),
            _fmt(sim_bb.get("break_index")),
            _bool_badge(sim_bb.get("significant", False), good_when_true=False),
        ])

    bp_res = data.get("bai_perron") or {}
    if "error" not in bp_res and bp_res:
        n_breaks = bp_res.get("optimal_breaks", 0)
        rows.append([
            "Bai-Perron (BIC)",
            f"{n_breaks} break(s) selected by BIC",
            "—",
            _esc(", ".join(str(b) for b in bp_res.get("break_indices", [])) or "none"),
            _bool_badge(n_breaks > 0, good_when_true=False),
        ])

    if not rows:
        err = data.get("error", "")
        return f'<div class="alert-warn">Structural break tests could not be run. {_esc(err)}</div>'

    out.append(_table(
        ["Test", "Statistic", "p-value", "Break Index / Indices", "Break Detected?"],
        rows,
        "Break detected = structural instability in the series"
    ))

    # ZA critical values
    za_cv = za.get("critical_values", {}) if za else {}
    if za_cv:
        out.append(_table(
            ["Significance Level", "ZA Critical Value"],
            [[_esc(k), _fmt(v, 4)] for k, v in za_cv.items()],
            "Zivot-Andrews critical values"
        ))

    # Bai-Perron segment detail table
    if bp_res and "segments" in bp_res and bp_res.get("optimal_breaks", 0) > 0:
        seg_rows = [
            [
                str(i + 1),
                _esc(s.get("date_start", str(s.get("start_index", "")))),
                _esc(s.get("date_end",   str(s.get("end_index", "")))),
                _fmt(s.get("n_obs")),
                _fmt(s.get("mean"), 4),
                _fmt(s.get("std"),  4),
            ]
            for i, s in enumerate(bp_res["segments"])
        ]
        out.append(_table(
            ["Segment", "Start", "End", "N Obs", "Mean", "Std Dev"],
            seg_rows,
            "Bai-Perron segment statistics"
        ))

    # Shift-in-means before/after detail
    if sim_bb:
        out.append(_kv([
            ("Break date",      _esc(str(sim_bb.get("break_date", sim_bb.get("break_index", "—"))))),
            ("Mean before",     _fmt(sim_bb.get("mean_before"), 4)),
            ("Mean after",      _fmt(sim_bb.get("mean_after"),  4)),
            ("Shift magnitude", _fmt(sim_bb.get("shift_magnitude"), 4)),
        ]))

    return "".join(out)


def _r_normality(data: dict) -> str:
    """Per-column normality test results (Jarque-Bera + Shapiro-Wilk)."""
    if not data or "error" in data:
        err = data.get("error", "") if isinstance(data, dict) else ""
        return f'<div class="alert-warn">Normality tests unavailable. {_esc(err)}</div>'

    rows = []
    for col, v in data.items():
        if not isinstance(v, dict) or "error" in v:
            continue
        jb  = v.get("jarque_bera", {})
        sw  = v.get("shapiro_wilk", {})

        def _norm_badge(is_normal):
            if is_normal is None:
                return "—"
            cls = "ok" if is_normal else "bad"
            return f'<span class="badge {cls}">{"Normal" if is_normal else "Non-normal"}</span>'

        sw_note = " *" if sw.get("subsampled") else ""
        rows.append([
            _esc(col),
            _fmt(jb.get("statistic"), 4),
            _pval_badge(jb.get("pvalue")),
            _norm_badge(jb.get("is_normal")),
            _fmt(sw.get("statistic"), 4),
            _pval_badge(sw.get("pvalue")),
            _norm_badge(sw.get("is_normal")),
            _esc(str(sw.get("n_used", "")) + sw_note),
        ])

    if not rows:
        return '<p class="muted">No columns with sufficient data for normality tests.</p>'

    out = [_table(
        ["Column", "JB Stat", "JB p-value", "JB Result",
         "SW Stat", "SW p-value", "SW Result", "SW n"],
        rows,
        "* SW subsampled to 5 000 obs for large columns"
    )]
    return "".join(out)


def _r_correlation(data: dict) -> str:
    """Pearson and Spearman correlation matrices (capped at 15 columns for readability)."""
    if not data or "error" in data:
        err = data.get("error", "") if isinstance(data, dict) else ""
        return f'<div class="alert-warn">Correlation analysis unavailable. {_esc(err)}</div>'

    cols   = data.get("columns", [])
    n_used = data.get("n_used", "?")
    LIMIT  = 15

    def _matrix_table(matrix: dict, label: str) -> str:
        display_cols = cols[:LIMIT]
        headers      = [""] + [_esc(str(c)) for c in display_cols]
        rows = []
        for row_col in display_cols:
            row_vals = matrix.get(row_col, {})
            row = [_esc(str(row_col))]
            for col_col in display_cols:
                v = row_vals.get(col_col)
                try:
                    fv = float(v)
                    # colour cells: strong positive = green, strong negative = red
                    if row_col == col_col:
                        cell = '<span style="color:#888">1.0000</span>'
                    elif abs(fv) >= 0.7:
                        colour = "#c0392b" if fv < 0 else "#27ae60"
                        cell = f'<span style="color:{colour};font-weight:bold">{fv:+.4f}</span>'
                    elif abs(fv) >= 0.4:
                        colour = "#e67e22" if fv < 0 else "#2980b9"
                        cell = f'<span style="color:{colour}">{fv:+.4f}</span>'
                    else:
                        cell = f"{fv:+.4f}"
                except (TypeError, ValueError):
                    cell = "—"
                row.append(cell)
            rows.append(row)
        cap = label + f" (n = {_fmt(n_used)} complete rows)"
        if len(cols) > LIMIT:
            cap += f" — showing first {LIMIT} of {len(cols)} columns"
        return _table(headers, rows, cap)

    out = [
        _matrix_table(data.get("pearson",  {}), "Pearson Correlation"),
        _matrix_table(data.get("spearman", {}), "Spearman Correlation"),
    ]
    return "".join(out)


def _r_arch_lm(data: dict) -> str:
    """ARCH-LM test result (single column)."""
    if not data or "error" in data:
        err = data.get("error", "") if isinstance(data, dict) else ""
        return f'<span class="muted">ARCH-LM: {_esc(err)}</span>'
    has_arch = data.get("has_arch_effects")
    cls      = "bad" if has_arch else "ok"
    label    = "ARCH effects detected" if has_arch else "No ARCH effects"
    rows     = [[
        _fmt(data.get("lm_statistic"), 4),
        _pval_badge(data.get("lm_pvalue")),
        _fmt(data.get("f_statistic"), 4),
        _pval_badge(data.get("f_pvalue")),
        _fmt(data.get("lags")),
        f'<span class="badge {cls}">{label}</span>',
    ]]
    return _table(
        ["LM Stat", "LM p-value", "F Stat", "F p-value", "Lags", "Conclusion"],
        rows,
        "ARCH-LM test for conditional heteroskedasticity (volatility clustering)"
    )


def _r_panel_balance(data: dict) -> str:
    if not data or "error" in data:
        err = data.get("error", "") if isinstance(data, dict) else ""
        return f'<div class="alert-warn">Panel balance check unavailable. {_esc(err)}</div>'

    is_bal = data.get("is_balanced")
    cls    = "ok" if is_bal else "bad"
    label  = "Balanced" if is_bal else "Unbalanced"
    ratio  = data.get("balance_ratio", 0)

    pairs = [
        ("Entities",              _fmt(data.get("n_entities"))),
        ("Time Periods",          _fmt(data.get("n_time_periods"))),
        ("Expected Observations", _fmt(data.get("expected_obs"))),
        ("Actual Observations",   _fmt(data.get("actual_obs"))),
        ("Missing Combinations",  _fmt(data.get("missing_combinations"))),
        ("Balance Ratio",         f"{ratio:.4f} (1.0 = fully balanced)"),
        ("Panel Type",            f'<span class="badge {cls}">{label}</span>'),
    ]
    return _kv(pairs)


def _r_panel_variance(data: dict) -> str:
    if not data or "error" in data:
        err = data.get("error", "") if isinstance(data, dict) else ""
        return f'<div class="alert-warn">Variance decomposition unavailable. {_esc(err)}</div>'

    rows = []
    for col, v in data.items():
        if not isinstance(v, dict):
            continue
        rows.append([
            _esc(col),
            _fmt(v.get("overall_variance"), 4),
            _fmt(v.get("between_variance"), 4),
            f'{v.get("between_pct", 0):.1f}%',
            _fmt(v.get("within_variance"), 4),
            f'{v.get("within_pct", 0):.1f}%',
        ])

    if not rows:
        return '<p class="muted">No numeric columns available for variance decomposition.</p>'

    return _table(
        ["Column", "Overall Var", "Between Var", "Between %", "Within Var", "Within %"],
        rows,
        "Between = variance of entity means; Within = mean of within-entity variances"
    )


def _r_panel(pr: Optional[dict], pf: Optional[dict]) -> str:
    out: List[str] = []

    if pr:
        out.append(_h3("Date Coverage"))
        pairs = [
            ("Global Start Date", _esc(str(pr.get("global_start", "—")))),
            ("Global End Date",   _esc(str(pr.get("global_end", "—")))),
        ]
        entity_stats = pr.get("entity_stats", {})
        if entity_stats:
            pairs.append(("Number of Entities", _fmt(len(entity_stats))))
        out.append(_kv(pairs))

        if entity_stats:
            rows = [
                [_esc(str(e)),
                 _esc(str(v.get("start", "—"))),
                 _esc(str(v.get("end", "—"))),
                 _fmt(v.get("n_obs"))]
                for e, v in list(entity_stats.items())[:100]
            ]
            cap = "Per-entity date coverage"
            if len(entity_stats) > 100:
                cap += f" (first 100 of {len(entity_stats)})"
            out.append(_table(["Entity", "Start Date", "End Date", "Observations"], rows, cap))

    if pf:
        out.append(_h3("Panel Frequency"))
        if "error" in pf:
            out.append(f'<p>{_esc(pf["error"])}</p>')
        elif "overridden_frequency" in pf:
            out.append(_kv([("Frequency (manual override)",
                              _esc(str(pf["overridden_frequency"])))]))
        else:
            rows = [[_esc(str(k)), _esc(str(v))] for k, v in pf.items()]
            if rows:
                out.append(_table(["Metric", "Value"], rows))

    return "".join(out)


# ---------------------------------------------------------------------------
# Executive summary
# ---------------------------------------------------------------------------

def _build_exec_summary(results: dict) -> str:
    info  = results.get("dataset_info", {})
    miss  = results.get("missing_duplicate", {}).get("missing", {})
    dups  = results.get("missing_duplicate", {}).get("duplicates", {})
    dtype = results.get("dataset_type", "—")

    miss_pct = float(miss.get("missing_percentage", 0) or 0)
    dup_pct  = float(dups.get("duplicate_percentage", 0) or 0)
    comp_pct = float(miss.get("complete_row_percentage", 0) or 0)

    def cls(val, low, mid, invert=False):
        if invert:
            return "good" if val >= low else ("warn" if val >= mid else "bad")
        return "good" if val <= low else ("warn" if val <= mid else "bad")

    rows_val = info.get("rows", "?")
    rows_fmt = f"{rows_val:,}" if isinstance(rows_val, int) else str(rows_val)

    cards = f"""
<div class="cards">
  <div class="card"><div class="card-lbl">File</div>
    <div class="card-val">{_esc(info.get("filename", "—"))}</div></div>
  <div class="card"><div class="card-lbl">Shape</div>
    <div class="card-val">{rows_fmt} &times; {info.get("columns", "?")}</div></div>
  <div class="card"><div class="card-lbl">Memory</div>
    <div class="card-val">{info.get("memory_mb", "—")} MB</div></div>
  <div class="card"><div class="card-lbl">Dataset Type</div>
    <div class="card-val">{_esc(str(dtype).replace("_", " ").title())}</div></div>
  <div class="card"><div class="card-lbl">Missing</div>
    <div class="card-val {cls(miss_pct, 1, 10)}">{miss_pct:.2f}%</div></div>
  <div class="card"><div class="card-lbl">Duplicates</div>
    <div class="card-val {cls(dup_pct, 0.5, 5)}">{dup_pct:.2f}%</div></div>
  <div class="card"><div class="card-lbl">Complete Rows</div>
    <div class="card-val {cls(comp_pct, 90, 70, invert=True)}">{comp_pct:.1f}%</div></div>
</div>"""

    alerts = ""
    # Prefer per-column results; fall back to flat keys (old format)
    _first_col = next(iter((results.get("columns") or {}).values()), {})
    stat = _first_col.get("stationarity") or results.get("stationarity") or {}
    if stat and "error" not in stat:
        adf  = stat.get("adf", {})
        kpss = stat.get("kpss_level", {})
        adf_r  = "Stationary" if adf.get("stationary") else "Non-stationary"
        kpss_r = "Stationary" if kpss.get("stationary") else "Non-stationary"
        try:
            adf_p  = f"{float(adf.get('pvalue', 0)):.4f}"
            kpss_p = f"{float(kpss.get('pvalue', 0)):.4f}"
        except (TypeError, ValueError):
            adf_p = kpss_p = "—"
        alerts += (f'<div class="qs neutral"><strong>Stationarity:</strong> '
                   f'ADF p = {adf_p} ({adf_r}) &bull; KPSS p = {kpss_p} ({kpss_r})</div>')

    sb = _first_col.get("structural_breaks") or results.get("structural_breaks") or {}
    if sb and "error" not in sb:
        breaks = []
        bb = (sb.get("chow") or {}).get("best_break") or {}
        if bb.get("significant"):
            breaks.append(f"Chow (F = {_fmt(bb.get('f_statistic'), 2)})")
        if (sb.get("cusum") or {}).get("rejects_stability"):
            breaks.append(f"CUSUM (max = {_fmt(sb['cusum'].get('max_cusum'), 3)})")
        if (sb.get("zivot_andrews") or {}).get("stationary_with_break"):
            breaks.append("Zivot-Andrews")
        sim_bb = (sb.get("shift_in_means") or {}).get("best_break") or {}
        if sim_bb.get("significant"):
            breaks.append(f"Shift in Means (Δμ = {_fmt(sim_bb.get('shift_magnitude'), 3)})")
        bp_res = sb.get("bai_perron") or {}
        if bp_res.get("optimal_breaks", 0) > 0:
            breaks.append(f"Bai-Perron ({bp_res['optimal_breaks']} break(s))")
        if breaks:
            alerts += (f'<div class="qs warn"><strong>Structural breaks detected:</strong> '
                       f'{"; ".join(breaks)}</div>')
        else:
            alerts += '<div class="qs ok"><strong>No significant structural breaks detected.</strong></div>'

    return (f'<div class="exec" id="summary"><h2>Executive Summary</h2>'
            f'{cards}{alerts}</div>')


# ---------------------------------------------------------------------------
# TOC
# ---------------------------------------------------------------------------

def _build_toc(entries: List[Tuple[str, str]]) -> str:
    items = "".join(f'<li><a href="#{a}">{_esc(t)}</a></li>' for t, a in entries)
    return f'<div class="toc"><h3>Contents</h3><ol>{items}</ol></div>'


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """
:root {
  --bg: #f8fafc; --fg: #1e293b; --accent: #2563eb; --accent2: #6d28d9;
  --ok: #16a34a; --warn: #d97706; --bad: #dc2626;
  --border: #e2e8f0; --card-bg: #ffffff; --muted: #64748b;
  --font: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  --mono: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: var(--font); color: var(--fg); background: var(--bg);
       padding: 28px 40px; line-height: 1.6; font-size: 14px; }
h1 { font-size: 22px; margin-bottom: 4px; color: var(--accent); font-weight: 700; }
h2 { font-size: 17px; margin: 28px 0 10px; padding-bottom: 5px;
     border-bottom: 2px solid var(--accent); color: var(--fg); font-weight: 600; }
h3 { font-size: 13px; margin: 18px 0 6px; color: var(--accent2); font-weight: 600;
     text-transform: uppercase; letter-spacing: 0.4px; }
.subtitle { color: var(--muted); margin-bottom: 18px; font-size: 12px; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }

/* TOC */
.toc { background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px;
       padding: 12px 18px; margin: 16px 0; max-width: 400px; }
.toc h3 { margin: 0 0 6px; font-size: 12px; color: var(--fg);
          text-transform: uppercase; letter-spacing: 0.4px; }
.toc ol { margin: 0; padding-left: 18px; }
.toc li { margin: 2px 0; font-size: 13px; }

/* Executive summary */
.exec { margin: 14px 0 24px; }
.exec h2 { margin-top: 0; }
.cards { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 10px; }
.card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px;
        padding: 10px 14px; min-width: 100px; text-align: center; }
.card-lbl { font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px;
            color: var(--muted); margin-bottom: 4px; }
.card-val { font-size: 15px; font-weight: 700; color: var(--fg); }
.card-val.good, .good { color: var(--ok); }
.card-val.warn, .warn { color: var(--warn); }
.card-val.bad,  .bad  { color: var(--bad); }
.qs { padding: 7px 12px; border-radius: 6px; margin: 5px 0; font-size: 13px;
      background: var(--card-bg); border: 1px solid var(--border); }
.qs.warn    { border-left: 4px solid var(--warn); }
.qs.ok      { border-left: 4px solid var(--ok); }
.qs.neutral { border-left: 4px solid var(--accent); }

/* Sections */
.section { margin-bottom: 32px; }

/* Tables */
.tbl-wrap { overflow-x: auto; margin: 8px 0 14px; border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,.07); }
table { border-collapse: collapse; width: 100%; font-size: 13px; background: var(--card-bg); }
thead th { background: var(--accent); color: #fff; padding: 8px 12px;
           text-align: left; font-weight: 600; white-space: nowrap; font-size: 12px; }
tbody td  { padding: 7px 12px; border-bottom: 1px solid var(--border); vertical-align: top; }
tbody tr:nth-child(even) td { background: #f1f5f9; }
tbody tr:hover td { background: #eff6ff; }
.kv thead th { background: #475569; }
.kv th { background: #f1f5f9; color: var(--fg); font-weight: 600; width: 34%;
         padding: 7px 12px; border-bottom: 1px solid var(--border);
         white-space: nowrap; vertical-align: top; }
.kv td { background: var(--card-bg) !important; }
.tbl-caption { font-size: 11px; color: var(--muted); margin: 2px 0 5px;
               font-style: italic; }

/* Badges */
.badge { display: inline-block; padding: 1px 7px; border-radius: 10px;
         font-size: 11px; font-weight: 600; white-space: nowrap; }
.badge.ok   { background: #dcfce7; color: var(--ok); }
.badge.warn { background: #fef3c7; color: var(--warn); }
.badge.bad  { background: #fee2e2; color: var(--bad); }

/* Alerts */
.alert-warn { background: #fef3c7; border-left: 4px solid var(--warn);
              padding: 9px 13px; border-radius: 4px; margin: 10px 0; font-size: 13px; }
.alert-info { background: #eff6ff; border-left: 4px solid #3b82f6;
              padding: 9px 13px; border-radius: 4px; margin: 10px 0; font-size: 13px; }

/* Plots */
.plot-grid { display: flex; flex-wrap: wrap; gap: 14px; margin: 14px 0; }
.plot-fig { margin: 0; flex: 1 1 400px; max-width: 100%; }
.plot-fig img { width: 100%; border: 1px solid var(--border); border-radius: 6px; }
.plot-fig figcaption { text-align: center; font-size: 11px; color: var(--muted);
                       margin-top: 4px; }

/* Per-column subsection label */
.subsec-lbl { font-size: 11px; font-weight: 700; text-transform: uppercase;
              letter-spacing: 0.5px; color: var(--muted); margin: 14px 0 4px; }
.muted { color: var(--muted); font-size: 12px; font-style: italic; margin: 4px 0; }
hr { border: none; border-top: 1px solid var(--border); margin: 24px 0; }

/* Footer */
.footer { margin-top: 32px; padding-top: 10px; border-top: 1px solid var(--border);
          color: var(--muted); font-size: 11px; }

@media print {
  body { padding: 10px; font-size: 11px; }
  .toc { break-after: page; }
  .plot-fig img { max-width: 100%; }
}
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_html_report(results: Dict[str, Any], output_path: str) -> None:
    """
    Generate a self-contained HTML validation report with embedded images.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plot_dirs: List[str] = results.get("plot_dirs", []) or []
    generated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fname = results.get("dataset_info", {}).get("filename", "")

    # ---- Build sections ----
    sections: List[Tuple[str, str, str]] = []  # (title, anchor, body)

    # 1. Dataset Overview
    info = results.get("dataset_info")
    if info:
        sections.append(("Dataset Overview", "dataset_overview",
                          _r_dataset_info(info)))

    # 2. Data Quality
    md = results.get("missing_duplicate")
    if md:
        sections.append(("Data Quality", "data_quality",
                          _r_missing_duplicate(md)))

    # 3. Descriptive Statistics + cross-sectional plots
    min_max  = results.get("min_max")
    avg      = results.get("average")
    med      = results.get("median")
    outliers = results.get("outliers")
    if any([min_max, avg, med, outliers]):
        body = _r_descriptive_stats(min_max, avg, med, outliers)
        xs_plots = _embed_plots(plot_dirs, "cross_sectional")
        if xs_plots:
            body += _h3("Visualisations") + xs_plots
        sections.append(("Descriptive Statistics", "descriptive_stats", body))

    # 4. Regression Diagnostics
    mc = results.get("multicollinearity")
    hk = results.get("heteroskedasticity")
    if mc or hk:
        sections.append(("Regression Diagnostics", "reg_diagnostics",
                          _r_regression_diagnostics(mc, hk)))

    # 4a. Normality Tests (cross-sectional)
    norm = results.get("normality")
    if norm and "error" not in norm:
        sections.append(("Normality Tests", "normality", _r_normality(norm)))

    # 4b. Correlation Analysis (cross-sectional)
    corr = results.get("correlation")
    if corr and "error" not in corr:
        sections.append(("Correlation Analysis", "correlation", _r_correlation(corr)))

    # 5. Time Series — Coverage
    rf  = results.get("range_frequency")
    fd  = results.get("frequency_details")
    mdf = results.get("missing_dates")
    if any([rf, fd, mdf]):
        sections.append(("Time Series — Coverage", "ts_coverage",
                          _r_ts_coverage(rf, fd, mdf)))

    # 6-9. Per-column time-series analysis (stationarity, LB, structural breaks, plots)
    columns = results.get("columns")
    if columns:
        col_body_parts: List[str] = []

        # Build a name→dir map once (O(m)) instead of rescanning per column (O(n×m))
        col_dir_map = {Path(d).name: d for d in (plot_dirs or [])}

        COLUMN_RENDER_LIMIT = 50
        col_items = list(columns.items())
        skipped = max(0, len(col_items) - COLUMN_RENDER_LIMIT)

        for col_name, col_data in col_items[:COLUMN_RENDER_LIMIT]:
            # Find this column's plot subdirectory via the pre-built map
            safe = "".join(c if c.isalnum() or c in "-_" else "_"
                           for c in str(col_name))[:60]
            col_plot_dir = col_dir_map.get(safe) or next(
                (d for d in (plot_dirs or []) if Path(d).name.startswith(safe)), None)
            col_plots = _embed_plots([col_plot_dir]) if col_plot_dir else ""

            col_html = _r_ts_column(col_name, col_data)
            if col_plots:
                col_html += col_plots
            col_body_parts.append(col_html)

        if skipped:
            col_body_parts.append(
                f'<div class="alert-info" style="padding:10px;margin:10px 0;">'
                f'{skipped} additional column(s) not rendered here to keep the report '
                f'manageable. Full per-column data is available in '
                f'<code>results.json</code>.</div>'
            )

        if col_body_parts:
            sections.append((
                "Time Series — Per Column Analysis",
                "ts_columns",
                "<hr>".join(col_body_parts)
            ))
    else:
        # Backward compat: flat keys from old single-column format
        stat_res = results.get("stationarity")
        if stat_res and "error" not in stat_res:
            sections.append(("Stationarity Tests", "stationarity",
                              _r_stationarity(stat_res)))
        lb = results.get("ljung_box")
        if lb and "error" not in lb:
            sections.append(("Autocorrelation — Ljung-Box", "ljung_box",
                              _r_ljung_box(lb)))
        sb = results.get("structural_breaks")
        if sb and "error" not in sb:
            sections.append(("Structural Break Tests", "structural_breaks",
                              _r_structural_breaks(sb)))
        ts_plots = _embed_plots(plot_dirs, "time_series")
        if ts_plots:
            sections.append(("Time Series Visualisations", "ts_plots", ts_plots))

    # 10. Panel Data + panel plots
    pr = results.get("panel_ranges")
    pf = results.get("panel_frequency")

    if pr or pf:
        body = _r_panel(pr, pf)
        panel_plots = _embed_plots(plot_dirs, "panel")
        if panel_plots:
            body += _h3("Panel Visualisations") + panel_plots
        sections.append(("Panel Data", "panel_data", body))

    # 10a. Panel — Entity Time-Series
    entity_ts = results.get("panel_entity_ts")

    if entity_ts:
        parts = []

        for ent, ent_res in entity_ts.items():
            parts.append(f"<h4>Entity: {ent}</h4>")

            # --- Coverage summary ---
            parts.append(
                _r_ts_coverage(
                    ent_res.get("range_frequency"),
                    ent_res.get("frequency_details"),
                    ent_res.get("missing_dates")
                )
            )

            # --- Per-column analysis with plots ---
            if "columns" in ent_res:
                for col_name, col_data in ent_res["columns"].items():

                    safe = "".join(
                        c if c.isalnum() or c in "-_" else "_" for c in str(col_name)
                    )[:60]

                    parts.append(f"<h5>Column: {col_name}</h5>")
                    parts.append(_r_ts_column(col_name, col_data))

                    # ---- ✅ Embed plots for this entity + column ----
                    col_dirs = [
                        d for d in plot_dirs
                        if f"panel_entity_ts/{ent}/{safe}" in d.replace("\\", "/")
                    ]

                    if col_dirs:
                        parts.append(_h3("Time-Series Plots"))
                        parts.append(_embed_plots(col_dirs))

        sections.append(
            ("Panel — Entity Time-Series", "panel_entity_ts", "\n".join(parts))
        )


    # 10a. Panel Balance
    pb = results.get("panel_balance")
    if pb:
        sections.append(("Panel Balance", "panel_balance", _r_panel_balance(pb)))

    # 10b. Panel Variance Decomposition
    pv = results.get("panel_variance")
    if pv and "error" not in pv:
        sections.append(("Panel Variance Decomposition", "panel_variance",
                          _r_panel_variance(pv)))

    # ---- TOC ----
    toc_entries = [("Executive Summary", "summary")] + [(t, a) for t, a, _ in sections]

    # ---- Assemble ----
    body_html = "\n".join(_section_html(t, a, b) for t, a, b in sections)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Validation Report — {_esc(fname)}</title>
<style>{_CSS}</style>
</head>
<body>

<h1>Data Validation Report</h1>
<div class="subtitle"><strong>{_esc(fname)}</strong> &mdash; generated {generated}</div>

{_build_toc(toc_entries)}

{_build_exec_summary(results)}

{body_html}

<div class="footer">Data Validator Pro &bull; {generated}</div>

</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
