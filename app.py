"""
app.py (Enhanced Pro Edition v3.0)
- Thread-safe GUI updates via message queue (fixes Tkinter race conditions)
- Working override dialogs from background thread (replaces TODO stub)
- Tabular preview with Treeview (replaces plain text)
- Log save/clear buttons
- Keyboard shortcuts: Ctrl+O, F5/Ctrl+R, Ctrl+Q
- Open Output Dir button
- Cancel unblocks waiting override dialogs
- Report existence check before opening
"""

import matplotlib
matplotlib.use("Agg")

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import json
import traceback
import webbrowser
import os
import threading
import queue
from datetime import datetime
from pathlib import Path
import csv

import pandas as pd

from main import StepwiseValidator, sanitize_for_json
from overrides.manager import OverrideManager
from overrides.dialogs import OverrideDialog
from overrides.registry import OverrideRegistry

try:
    import ttkbootstrap as ttk_bootstrap
    HAVE_BOOTSTRAP = True
except ImportError:
    HAVE_BOOTSTRAP = False


class ValidatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Validator Pro - Advanced Dataset Analysis")
        self.root.geometry("1500x950")
        self.root.minsize(900, 600)

        self.override_mgr = OverrideManager()
        self.override_registry = OverrideRegistry()

        # UI state
        self.file_path = tk.StringVar()
        self.data_type = tk.StringVar(value="cross_sectional")
        self.output_base_dir = tk.StringVar(value="results")
        self.last_report_path = None
        self.selected_sheet = None

        # Threading
        self.validation_queue = queue.Queue()
        self.validation_thread = None
        self.is_running = False
        self.df_preview = None

        # Override dialog synchronization (worker blocks until user responds)
        self._dialog_event = threading.Event()
        self._dialog_result = {}

        # Caching
        self.last_validation_results = None
        self.run_history = []
        self._load_run_history()

        self._build_ui()
        self._bind_shortcuts()
        self._check_queue()

    # ============================================================
    # Keyboard Shortcuts
    # ============================================================
    def _bind_shortcuts(self):
        self.root.bind("<Control-o>", lambda e: self._browse_file())
        self.root.bind("<Control-r>", lambda e: self._start_validation_thread())
        self.root.bind("<Control-q>", lambda e: self.root.quit())
        self.root.bind("<F5>", lambda e: self._start_validation_thread())

    # ============================================================
    # Run History
    # ============================================================
    def _load_run_history(self):
        history_file = Path("run_history.json")
        if history_file.exists():
            try:
                with open(history_file, encoding="utf-8") as f:
                    self.run_history = json.load(f)[-10:]
            except Exception:
                self.run_history = []

    def _save_run_history(self, run_info):
        self.run_history.append(run_info)
        self.run_history = self.run_history[-10:]
        try:
            with open("run_history.json", "w", encoding="utf-8") as f:
                json.dump(self.run_history, f, indent=2)
        except Exception:
            pass

    # ============================================================
    # Main UI Builder
    # ============================================================
    def _build_ui(self):
        self._build_banner()

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=5, pady=5)

        self.tab_input = ttk.Frame(notebook)
        self.tab_preview = ttk.Frame(notebook)
        self.tab_log = ttk.Frame(notebook)
        self.tab_results = ttk.Frame(notebook)
        self.tab_history = ttk.Frame(notebook)
        self.tab_settings = ttk.Frame(notebook)

        notebook.add(self.tab_input, text="Input")
        notebook.add(self.tab_preview, text="Preview")
        notebook.add(self.tab_log, text="Log")
        notebook.add(self.tab_results, text="Results")
        notebook.add(self.tab_history, text="History")
        notebook.add(self.tab_settings, text="Settings")

        self._build_input_tab()
        self._build_preview_tab()
        self._build_log_tab()
        self._build_results_tab()
        self._build_history_tab()
        self._build_settings_tab()

    def _build_banner(self):
        banner = ttk.Frame(self.root)
        banner.pack(fill="x", padx=5, pady=5)

        ttk.Label(banner, text="Data Validator", font=("Helvetica", 16, "bold")).pack(side="left", padx=10)
        ttk.Label(
            banner,
            text="Ctrl+O: Open  |  F5/Ctrl+R: Run  |  Ctrl+Q: Quit",
            font=("Arial", 8),
            foreground="gray"
        ).pack(side="left", padx=20)

        self.status_label = ttk.Label(banner, text="✓ Ready", foreground="green")
        self.status_label.pack(side="right", padx=20)

        self.progress = ttk.Progressbar(self.root, mode="indeterminate", length=200)
        self.progress.pack(fill="x", padx=5, pady=2)

    # ============================================================
    # Input Tab
    # ============================================================
    def _build_input_tab(self):
        canvas = tk.Canvas(self.tab_input, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.tab_input, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        self._build_file_section(scrollable_frame)
        self._build_columns_section(scrollable_frame)
        self._build_options_section(scrollable_frame)
        self._build_action_buttons(scrollable_frame)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _build_file_section(self, parent):
        frame = ttk.LabelFrame(parent, text="1️Select Dataset File", padding=10)
        frame.pack(fill="x", padx=10, pady=5)

        file_frame = ttk.Frame(frame)
        file_frame.pack(fill="x", pady=5)

        ttk.Label(file_frame, text="File Path:", font=("Arial", 10)).pack(side="left", padx=5)
        ttk.Entry(file_frame, textvariable=self.file_path, width=70).pack(side="left", padx=5, fill="x", expand=True)
        ttk.Button(file_frame, text="Browse", command=self._browse_file, width=15).pack(side="left", padx=5)

    def _build_columns_section(self, parent):
        frame = ttk.LabelFrame(parent, text="2️Detected Columns", padding=10)
        frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(frame, text="Dataset Columns (detected automatically):", font=("Arial", 10)).pack(anchor="w", pady=(0, 5))

        lb_frame = ttk.Frame(frame)
        lb_frame.pack(fill="x", expand=True)

        scrollbar = ttk.Scrollbar(lb_frame)
        scrollbar.pack(side="right", fill="y")

        self.column_list = tk.Listbox(lb_frame, height=6, yscrollcommand=scrollbar.set)
        self.column_list.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.column_list.yview)

    def _build_options_section(self, parent):
        frame = ttk.LabelFrame(parent, text="3️Data Configuration", padding=10)
        frame.pack(fill="x", padx=10, pady=5)

        type_frame = ttk.Frame(frame)
        type_frame.pack(fill="x", pady=5)
        ttk.Label(type_frame, text="Dataset Type:", font=("Arial", 10), width=20).pack(side="left", padx=5)
        self.dataset_type_combo = ttk.Combobox(
            type_frame,
            values=["cross_sectional", "time_series", "panel"],
            textvariable=self.data_type,
            state="readonly",
            width=30
        )
        self.dataset_type_combo.pack(side="left", padx=5)
        self.dataset_type_combo.bind("<<ComboboxSelected>>", self._on_type_change)

        date_frame = ttk.Frame(frame)
        date_frame.pack(fill="x", pady=5)
        ttk.Label(date_frame, text="Date Column (optional):", font=("Arial", 10), width=20).pack(side="left", padx=5)
        self.date_column_dropdown = ttk.Combobox(date_frame, values=[], state="disabled", width=30)
        self.date_column_dropdown.pack(side="left", padx=5)

        entity_frame = ttk.Frame(frame)
        entity_frame.pack(fill="x", pady=5)
        ttk.Label(entity_frame, text="Entity Column (optional):", font=("Arial", 10), width=20).pack(side="left", padx=5)
        self.entity_column_dropdown = ttk.Combobox(entity_frame, values=[], state="disabled", width=30)
        self.entity_column_dropdown.pack(side="left", padx=5)

        output_frame = ttk.Frame(frame)
        output_frame.pack(fill="x", pady=5)
        ttk.Label(output_frame, text="Output Directory:", font=("Arial", 10), width=20).pack(side="left", padx=5)
        ttk.Entry(output_frame, textvariable=self.output_base_dir, width=40).pack(side="left", padx=5, fill="x", expand=True)
        ttk.Button(output_frame, text="Browse", command=self._browse_output_dir, width=12).pack(side="left", padx=5)

    def _build_action_buttons(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill="x", padx=10, pady=15)

        ttk.Button(frame, text="Run Validation (F5)", command=self._start_validation_thread, width=25).pack(side="left", padx=5)
        ttk.Button(frame, text="Cancel", command=self._cancel_validation, width=20).pack(side="left", padx=5)
        ttk.Button(frame, text="Open Report", command=self._open_report, width=20).pack(side="left", padx=5)
        ttk.Button(frame, text="Open Output Dir", command=self._open_output_dir, width=20).pack(side="left", padx=5)

    def _open_output_dir(self):
        out = self.output_base_dir.get() or "results"
        path = Path(out)
        if path.exists():
            os.startfile(str(path.resolve()))
        else:
            messagebox.showinfo("Output Directory", f"Directory does not exist yet:\n{path.resolve()}")

    # ============================================================
    # Preview Tab  — tabular Treeview instead of plain text
    # ============================================================
    def _build_preview_tab(self):
        frame = ttk.LabelFrame(self.tab_preview, text="Data Preview", padding=10)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        top = ttk.Frame(frame)
        top.pack(fill="x", pady=5)

        self.preview_info_label = ttk.Label(top, text="Select a file and click Load Preview", font=("Arial", 9))
        self.preview_info_label.pack(side="left", padx=5)
        ttk.Button(top, text="Load Preview Data", command=self._show_preview).pack(side="right", padx=5)

        tree_frame = ttk.Frame(frame)
        tree_frame.pack(fill="both", expand=True)

        h_scroll = ttk.Scrollbar(tree_frame, orient="horizontal")
        v_scroll = ttk.Scrollbar(tree_frame, orient="vertical")

        self.preview_tree = ttk.Treeview(
            tree_frame,
            yscrollcommand=v_scroll.set,
            xscrollcommand=h_scroll.set,
            height=20
        )

        h_scroll.config(command=self.preview_tree.xview)
        v_scroll.config(command=self.preview_tree.yview)

        h_scroll.pack(side="bottom", fill="x")
        v_scroll.pack(side="right", fill="y")
        self.preview_tree.pack(side="left", fill="both", expand=True)

    def _show_preview(self):
        if not self.file_path.get():
            messagebox.showwarning("Preview", "Please select a file first")
            return

        try:
            path = self.file_path.get()
            if path.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(path, sheet_name=self.selected_sheet, nrows=50)
            else:
                df = pd.read_csv(path, nrows=50)

            self.preview_tree.delete(*self.preview_tree.get_children())
            self.preview_tree["columns"] = list(df.columns)
            self.preview_tree["show"] = "headings"

            for col in df.columns:
                self.preview_tree.heading(col, text=col)
                self.preview_tree.column(col, width=max(80, min(200, len(col) * 10)), anchor="w")

            for _, row in df.iterrows():
                self.preview_tree.insert("", "end", values=[str(v) for v in row])

            self.preview_info_label.config(text=f"Showing {len(df)} rows × {len(df.columns)} columns")
            self.log(f"Preview loaded: {df.shape[0]} rows × {df.shape[1]} columns", "success")
        except Exception as e:
            messagebox.showerror("Error", f"Could not preview data:\n\n{e}")

    # ============================================================
    # Log Tab
    # ============================================================
    def _build_log_tab(self):
        frame = ttk.LabelFrame(self.tab_log, text="Validation Log", padding=10)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill="x", pady=(0, 5))
        ttk.Button(btn_frame, text="Clear Log", command=self._clear_log, width=15).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Save Log", command=self._save_log, width=15).pack(side="left", padx=5)

        self.log_text = scrolledtext.ScrolledText(frame, height=25, width=150, font=("Courier", 9))
        self.log_text.pack(fill="both", expand=True)

        self.log_text.tag_config("info", foreground="white")
        self.log_text.tag_config("success", foreground="lightgreen")
        self.log_text.tag_config("warning", foreground="yellow")
        self.log_text.tag_config("error", foreground="salmon")

    def _clear_log(self):
        self.log_text.delete("1.0", "end")

    def _save_log(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All", "*.*")]
        )
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(self.log_text.get("1.0", "end"))
                messagebox.showinfo("Save Log", f"Log saved to:\n{path}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save log:\n{e}")

    def log(self, msg, tag="info"):
        """Append a coloured message to the log (must be called from main thread)."""
        self.log_text.insert("end", msg + "\n", tag)
        self.log_text.see("end")

    # ============================================================
    # Results Tab
    # ============================================================
    def _build_results_tab(self):
        frame = ttk.LabelFrame(self.tab_results, text="Validation Results", padding=10)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        tree_frame = ttk.Frame(frame)
        tree_frame.pack(fill="both", expand=True, pady=10)

        h_scroll = ttk.Scrollbar(tree_frame, orient="horizontal")
        v_scroll = ttk.Scrollbar(tree_frame, orient="vertical")

        columns = ("Metric", "Value")
        self.results_tree = ttk.Treeview(
            tree_frame, columns=columns, height=20,
            show="tree headings",
            yscrollcommand=v_scroll.set,
            xscrollcommand=h_scroll.set,
        )
        h_scroll.config(command=self.results_tree.xview)
        v_scroll.config(command=self.results_tree.yview)

        h_scroll.pack(side="bottom", fill="x")
        v_scroll.pack(side="right", fill="y")
        self.results_tree.pack(side="left", fill="both", expand=True)

        self.results_tree.heading("#0", text="Category / Column")
        self.results_tree.heading("Metric", text="Metric")
        self.results_tree.heading("Value", text="Value")
        self.results_tree.column("#0", width=220, minwidth=120)
        self.results_tree.column("Metric", width=200, minwidth=100)
        self.results_tree.column("Value", width=500, minwidth=150)

        export_frame = ttk.Frame(frame)
        export_frame.pack(fill="x", pady=10)
        ttk.Button(export_frame, text="Export JSON", command=self._export_json, width=20).pack(side="left", padx=5)
        ttk.Button(export_frame, text="Export CSV", command=self._export_csv, width=20).pack(side="left", padx=5)
        ttk.Button(export_frame, text="Export All", command=self._export_all, width=20).pack(side="left", padx=5)

    def _populate_results_tree(self, results):
        """Populate results tree view. Must be called from main thread."""
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        def _insert(parent, key, value, depth=0):
            key_str = str(key)
            if depth >= 3:
                self.results_tree.insert(parent, "end", values=(key_str[:80], str(value)[:300]))
                return
            if isinstance(value, dict):
                node = self.results_tree.insert(parent, "end", text=key_str, open=(depth == 0))
                items = list(value.items())
                for k, v in items[:200]:
                    _insert(node, k, v, depth + 1)
                if len(items) > 200:
                    self.results_tree.insert(node, "end", values=(f"… {len(items) - 200} more items", ""))
            elif isinstance(value, (list, tuple)):
                node = self.results_tree.insert(parent, "end", text=key_str, open=(depth == 0))
                for i, item in enumerate(list(value)[:100]):
                    _insert(node, f"[{i}]", item, depth + 1)
                if len(value) > 100:
                    self.results_tree.insert(node, "end", values=(f"… {len(value) - 100} more items", ""))
            else:
                self.results_tree.insert(parent, "end", values=(key_str[:80], str(value)[:300]))

        for key, value in results.items():
            if key == "plot_dirs":
                continue
            _insert("", key, value, depth=0)

    def _export_json(self):
        if not self.last_validation_results:
            messagebox.showwarning("Export", "No results to export. Run validation first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(sanitize_for_json(self.last_validation_results), f, indent=2)
                messagebox.showinfo("Export", f"Results exported to:\n{path}")
                self.log(f"✓ Exported JSON: {path}", "success")
            except Exception as e:
                messagebox.showerror("Error", f"Could not export:\n\n{e}")

    def _export_csv(self):
        if not self.last_validation_results:
            messagebox.showwarning("Export", "No results to export. Run validation first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if path:
            try:
                flat = self._flatten_results(self.last_validation_results)
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Key", "Value"])
                    for k, v in flat:
                        writer.writerow([k, str(v)])
                messagebox.showinfo("Export", f"Results exported to:\n{path}")
                self.log(f"✓ Exported CSV: {path}", "success")
            except Exception as e:
                messagebox.showerror("Error", f"Could not export:\n\n{e}")

    def _export_all(self):
        if not self.last_validation_results:
            messagebox.showwarning("Export", "No results to export. Run validation first.")
            return
        folder = filedialog.askdirectory(title="Select export folder")
        if folder:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                json_path = os.path.join(folder, f"results_{timestamp}.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(sanitize_for_json(self.last_validation_results), f, indent=2)

                csv_path = os.path.join(folder, f"results_{timestamp}.csv")
                flat = self._flatten_results(self.last_validation_results)
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Key", "Value"])
                    for k, v in flat:
                        writer.writerow([k, str(v)])

                messagebox.showinfo("Export", f"All formats exported to:\n{folder}")
                self.log(f"✓ Exported all formats to: {folder}", "success")
            except Exception as e:
                messagebox.showerror("Error", f"Could not export:\n\n{e}")

    def _flatten_results(self, obj, prefix="", max_depth=3, current_depth=0):
        """Flatten nested results dict for CSV export."""
        items = []
        if current_depth >= max_depth:
            return [(prefix, str(obj))]
        if isinstance(obj, dict):
            for k, v in list(obj.items())[:50]:
                new_prefix = f"{prefix}.{k}" if prefix else k
                items.extend(self._flatten_results(v, new_prefix, max_depth, current_depth + 1))
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(list(obj)[:10]):
                new_prefix = f"{prefix}[{i}]"
                items.extend(self._flatten_results(v, new_prefix, max_depth, current_depth + 1))
        else:
            items.append((prefix, obj))
        return items

    # ============================================================
    # History Tab
    # ============================================================
    def _build_history_tab(self):
        frame = ttk.LabelFrame(self.tab_history, text="Validation Run History", padding=10)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        ttk.Label(frame, text="Recent validation runs (last 10):", font=("Arial", 10)).pack(anchor="w", pady=(0, 5))

        list_frame = ttk.Frame(frame)
        list_frame.pack(fill="both", expand=True, pady=5)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")

        self.history_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, height=15, font=("Courier", 9))
        self.history_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.history_listbox.yview)

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill="x", pady=10)
        ttk.Button(btn_frame, text="Load Selected", command=self._load_history_run, width=20).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Clear History", command=self._clear_history, width=20).pack(side="left", padx=5)

        self._refresh_history()

    def _refresh_history(self):
        """Refresh history listbox. Must be called from main thread."""
        self.history_listbox.delete(0, "end")
        if not self.run_history:
            self.history_listbox.insert("end", "(No history yet)")
        else:
            for run in reversed(self.run_history):
                timestamp = run.get("timestamp", "Unknown")[:10]
                file = run.get("file", "Unknown")[:50]
                data_type = run.get("type", "?")
                self.history_listbox.insert("end", f"{timestamp} | {data_type:15} | {file}")

    def _load_history_run(self):
        sel = self.history_listbox.curselection()
        if sel:
            idx = sel[0]
            if idx < len(self.run_history):
                run = self.run_history[-(idx + 1)]
                self.last_report_path = run.get("report")
                self._open_report()

    def _clear_history(self):
        if messagebox.askyesno("Confirm", "Clear all run history?"):
            self.run_history = []
            try:
                Path("run_history.json").unlink()
            except Exception:
                pass
            self._refresh_history()
            self.log("✓ History cleared", "success")

    # ============================================================
    # Settings Tab
    # ============================================================
    def _build_settings_tab(self):
        frame = ttk.Frame(self.tab_settings, padding=10)
        frame.pack(fill="both", expand=True)

        cache_frame = ttk.LabelFrame(frame, text="Cache & Overrides Management", padding=10)
        cache_frame.pack(fill="x", padx=5, pady=5)

        ttk.Button(cache_frame, text="Clear Overrides", command=self._clear_overrides, width=30).pack(anchor="w", padx=5, pady=5)
        ttk.Button(cache_frame, text="Reset Application", command=self._reset_application, width=30).pack(anchor="w", padx=5, pady=5)

        info_frame = ttk.LabelFrame(frame, text="Output Directory Structure", padding=10)
        info_frame.pack(fill="x", padx=5, pady=5)

        info_text = scrolledtext.ScrolledText(info_frame, height=8, width=80, font=("Courier", 9))
        info_text.pack(fill="both", expand=True)
        info_text.insert("1.0", """Output files are organized as:
<base_directory>/
├── <YYYY-MM-DD_HH-MM-SS>/
│   ├── report.html                 # Main HTML report
│   ├── results.json                # Full results (JSON)
│   ├── csv/
│   │   └── *.csv                   # Result tables
│   └── *_plots/
│       └── *.png                   # Generated visualizations

Examples:
• results/2026-02-21_14-35-22/
• results/2026-02-21_14-35-22/report.html
• results/2026-02-21_14-35-22/cross_sectional_plots/""")
        info_text.config(state="disabled")

        about_frame = ttk.LabelFrame(frame, text="About Data Validator Pro", padding=10)
        about_frame.pack(fill="both", expand=True, padx=5, pady=5)

        about_text = scrolledtext.ScrolledText(about_frame, height=10, width=80, font=("Courier", 9))
        about_text.pack(fill="both", expand=True)
        about_text.insert("1.0", """Data Validator Pro v3.0
Advanced Dataset Validation & Analysis Tool

Keyboard Shortcuts:
  Ctrl+O          Open file browser
  F5 / Ctrl+R     Run validation
  Ctrl+Q          Quit

Supported Data Types:
  ✓ Cross-sectional datasets
  ✓ Time-series data
  ✓ Panel datasets (longitudinal)

Key Features:
  ✓ Thread-safe GUI (no freezes, safe override dialogs)
  ✓ Tabular data preview (50 rows via Treeview)
  ✓ Save/Clear validation log
  ✓ Run history and caching (last 10 runs)
  ✓ Multiple export formats (JSON, CSV, HTML)
  ✓ Memory-optimized for large datasets
  ✓ Automatic encoding/delimiter detection
  ✓ Support for Excel multi-sheet files

© 2026 - Enhanced Edition v3.0""")
        about_text.config(state="disabled")

    # ============================================================
    # File Operations
    # ============================================================
    def _browse_file(self):
        path = filedialog.askopenfilename(
            filetypes=[
                ("All Supported", "*.csv;*.xlsx;*.xls;*.json"),
                ("CSV Files", "*.csv"),
                ("Excel Files", "*.xlsx;*.xls"),
                ("JSON Files", "*.json"),
                ("All Files", "*.*")
            ]
        )
        if not path:
            return
        self.file_path.set(path)
        self._load_file_preview(path)

    def _load_file_preview(self, path):
        try:
            if path.lower().endswith((".xlsx", ".xls")):
                xls = pd.ExcelFile(path)
                sheets = xls.sheet_names
                sheet = self._ask_sheet_selection(sheets) if len(sheets) > 1 else sheets[0]
                if not sheet:
                    return
                self.df_preview = pd.read_excel(path, sheet_name=sheet, nrows=0, engine="openpyxl")
                self.selected_sheet = sheet
            else:
                self.df_preview = pd.read_csv(path, nrows=0)

            cols = list(self.df_preview.columns)
            self.column_list.delete(0, "end")
            for c in cols:
                self.column_list.insert("end", c)

            self.date_column_dropdown["values"] = cols
            self.entity_column_dropdown["values"] = cols

            self._update_status(f"✓ {Path(path).name} loaded ({len(cols)} columns)")
            self.log(f"✓ File loaded: {Path(path).name} ({len(cols)} columns)", "success")

        except Exception as e:
            self.log(f"✗ Error loading file: {e}", "error")
            messagebox.showerror("Error", f"Could not read file.\n\n{e}")

    def _ask_sheet_selection(self, sheets):
        win = tk.Toplevel(self.root)
        win.title("Select Worksheet")
        win.geometry("320x180")
        win.grab_set()

        ttk.Label(win, text="Multiple sheets found.\nSelect one to use:", font=("Arial", 10)).pack(pady=10)
        selected = tk.StringVar(value=sheets[0])
        ttk.Combobox(win, values=sheets, textvariable=selected, state="readonly", width=30).pack(pady=10)

        def confirm():
            win.destroy()

        ttk.Button(win, text="OK", command=confirm).pack(pady=10)
        win.wait_window()
        return selected.get()

    def _browse_output_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.output_base_dir.set(d)

    # ============================================================
    # Validation (Threading)
    # ============================================================
    def _start_validation_thread(self):
        if not self.file_path.get():
            messagebox.showwarning("Validation", "Please select a file first")
            return
        if self.is_running:
            messagebox.showinfo("Info", "Validation already running")
            return

        self.is_running = True
        self._update_status("⏳ Validation in progress...")
        self.progress.start()
        self.log_text.delete("1.0", "end")
        self.log("=" * 60)
        self.log(f"Starting validation: {Path(self.file_path.get()).name}")
        self.log(f"Data type: {self.data_type.get()}")
        self.log("=" * 60)

        self.validation_thread = threading.Thread(target=self._run_validation_worker, daemon=True)
        self.validation_thread.start()

    def _run_validation_worker(self):
        """Worker thread. All GUI writes go through the queue for thread safety."""
        try:
            validator = StepwiseValidator(
                self.file_path.get(),
                self.data_type.get(),
                self.override_mgr,
                self._queue_log,
                self._handle_override_dialog
            )

            validator.output_dir_base = self.output_base_dir.get()
            validator.selected_sheet = self.selected_sheet

            if self.date_column_dropdown.get():
                validator.override_mgr.set("date_column", self.date_column_dropdown.get())
            if self.entity_column_dropdown.get():
                validator.override_mgr.set("entity_column", self.entity_column_dropdown.get())

            results = validator.run_all_steps()
            report_path = validator.generate_html_report()

            self.last_report_path = report_path
            self.last_validation_results = results

            run_info = {
                "timestamp": datetime.now().isoformat(),
                "file": Path(self.file_path.get()).name,
                "type": self.data_type.get(),
                "report": report_path
            }
            self._save_run_history(run_info)

            # Queue GUI updates (thread-safe — never update Tkinter directly from here)
            self.validation_queue.put(("populate_tree", results))
            self.validation_queue.put(("refresh_history", None))

            self._queue_log("=" * 60)
            self._queue_log("✓ Validation completed successfully!", "success")
            self._queue_log(f"Report: {report_path}", "success")
            self._queue_log("=" * 60)
            self.validation_queue.put(("done", None))

        except Exception as e:
            self._queue_log("=" * 60, "error")
            self._queue_log(f"✗ Validation failed: {e}", "error")
            self._queue_log(traceback.format_exc(), "error")
            self._queue_log("=" * 60, "error")
            self.validation_queue.put(("error", str(e)))

    def _queue_log(self, msg, tag="info"):
        """Queue a log message from a worker thread."""
        self.validation_queue.put(("log", (msg, tag)))

    def _handle_override_dialog(self, dialog_spec):
        """
        Show an override dialog from the worker thread.

        Posts a 'show_dialog' message to the main thread, then blocks the
        worker on a threading.Event until the user clicks OK/Cancel.
        Thread-safe: never calls Tkinter directly.
        """
        self._dialog_event.clear()
        self._dialog_result = {}
        self.validation_queue.put(("show_dialog", dialog_spec))
        self._dialog_event.wait()   # worker blocks here
        return self._dialog_result

    def _show_override_dialog_in_main(self, dialog_spec):
        """Render the override dialog in the main thread and unblock the worker."""
        win = tk.Toplevel(self.root)
        title = dialog_spec.get("title", "Override Required")
        win.title(title)
        win.grab_set()
        win.geometry("420x320")
        win.resizable(False, False)

        ttk.Label(win, text=title, font=("Arial", 12, "bold")).pack(pady=10)

        fields = dialog_spec.get("fields", {})
        field_vars = {}

        for field_name, field_cfg in fields.items():
            ftype = field_cfg.get("type", "text")
            label = field_cfg.get("label", field_name)

            row = ttk.Frame(win)
            row.pack(fill="x", padx=20, pady=5)
            ttk.Label(row, text=label, width=30, anchor="w").pack(side="left")

            if ftype == "dropdown":
                var = tk.StringVar()
                opts = field_cfg.get("options", [])
                if opts:
                    var.set(opts[0])
                ttk.Combobox(row, textvariable=var, values=opts, state="readonly", width=20).pack(side="left", padx=5)
            else:
                var = tk.StringVar()
                ttk.Entry(row, textvariable=var, width=25).pack(side="left", padx=5)

            field_vars[field_name] = var

        def confirm():
            self._dialog_result = {k: v.get() for k, v in field_vars.items()}
            win.destroy()
            self._dialog_event.set()

        def cancel():
            self._dialog_result = {}
            win.destroy()
            self._dialog_event.set()

        btn_row = ttk.Frame(win)
        btn_row.pack(pady=15)
        ttk.Button(btn_row, text="OK", command=confirm, width=15).pack(side="left", padx=5)
        ttk.Button(btn_row, text="Cancel", command=cancel, width=15).pack(side="left", padx=5)

    def _cancel_validation(self):
        if self.is_running:
            self.is_running = False
            self._update_status("⏹ Validation cancelled")
            self.progress.stop()
            self.log("⏹ Validation cancelled by user", "warning")
            # Unblock any dialog the worker might be waiting on
            self._dialog_event.set()

    def _check_queue(self):
        """Drain the queue and dispatch GUI updates (runs in main thread every 100 ms)."""
        try:
            while True:
                msg_type, msg_data = self.validation_queue.get_nowait()

                if msg_type == "log":
                    msg_text, tag = msg_data
                    self.log(msg_text, tag)
                elif msg_type == "done":
                    self.is_running = False
                    self.progress.stop()
                    self._update_status("✓ Validation complete")
                elif msg_type == "error":
                    self.is_running = False
                    self.progress.stop()
                    self._update_status("✗ Validation failed")
                elif msg_type == "populate_tree":
                    self._populate_results_tree(msg_data)
                elif msg_type == "refresh_history":
                    self._refresh_history()
                elif msg_type == "show_dialog":
                    self._show_override_dialog_in_main(msg_data)

        except queue.Empty:
            pass

        self.root.after(100, self._check_queue)

    # ============================================================
    # Report Management
    # ============================================================
    def _open_report(self):
        if not self.last_report_path:
            messagebox.showwarning("Report", "No report available. Run validation first.")
            return
        report = Path(self.last_report_path).resolve()
        if not report.exists():
            messagebox.showwarning("Report", f"Report file not found:\n{report}")
            return
        try:
            os.startfile(str(report))
            self.log(f"✓ Report opened: {report}", "success")
        except Exception as e:
            messagebox.showerror("Error", f"Could not open report:\n\n{e}")

    # ============================================================
    # UI Helpers
    # ============================================================
    def _on_type_change(self, event=None):
        t = self.data_type.get()
        if t == "cross_sectional":
            self.date_column_dropdown.config(state="disabled")
            self.entity_column_dropdown.config(state="disabled")
        elif t == "time_series":
            self.date_column_dropdown.config(state="readonly")
            self.entity_column_dropdown.config(state="disabled")
        elif t == "panel":
            self.date_column_dropdown.config(state="readonly")
            self.entity_column_dropdown.config(state="readonly")

    def _update_status(self, status):
        self.status_label.config(text=status)

    def _clear_overrides(self):
        if messagebox.askyesno("Confirm", "Clear all overrides?"):
            self.override_mgr.clear()
            self.log("✓ Overrides cleared", "success")

    def _reset_application(self):
        if messagebox.askyesno("Confirm", "Reset application to initial state?"):
            self.file_path.set("")
            self.column_list.delete(0, "end")
            self.data_type.set("cross_sectional")
            self.dataset_type_combo.set("cross_sectional")
            self.date_column_dropdown.set("")
            self.entity_column_dropdown.set("")
            self.output_base_dir.set("results")
            self.override_mgr.clear()

            self.log_text.delete("1.0", "end")
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)

            self.preview_tree.delete(*self.preview_tree.get_children())
            self.preview_tree["columns"] = []
            self.preview_info_label.config(text="Select a file and click Load Preview")

            self._update_status("✓ Ready")
            self.log("Application reset to initial state", "success")


def main():
    if HAVE_BOOTSTRAP:
        root = ttk_bootstrap.Window(themename="darkly")
    else:
        root = tk.Tk()

    app = ValidatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
