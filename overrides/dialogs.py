
"""
dialogs.py

Tkinter popup dialog that displays one or more override input fields.
Supports:

- text fields
- dropdown fields (combobox)
- multiselect fields

Used by the Tkinter GUI to capture manual override values.
"""

import tkinter as tk
from tkinter import ttk


class OverrideDialog(tk.Toplevel):

    def __init__(self, parent, dialog_definition):
        super().__init__(parent)

        self.title(dialog_definition.get("title", "Manual Override"))
        self.geometry("420x300")
        self.resizable(False, False)
        self.result = {}

        fields = dialog_definition.get("fields", {})

        # Build UI
        row = 0
        self.inputs = {}  # maps key -> widget or widget list

        for key, field in fields.items():
            label_text = field.get("label", key)
            field_type = field.get("type", "text")

            tk.Label(self, text=label_text, anchor="w", font=("Arial", 11))\
                .grid(row=row, column=0, sticky="w", padx=10, pady=5)

            if field_type == "text":
                var = tk.StringVar()
                entry = tk.Entry(self, textvariable=var, width=40)
                entry.grid(row=row, column=1, padx=5)
                self.inputs[key] = var

            elif field_type == "dropdown":
                options = field.get("options", [])
                var = tk.StringVar()
                combo = ttk.Combobox(self, textvariable=var, values=options,
                                     state="readonly", width=37)
                combo.grid(row=row, column=1, padx=5)
                if options:
                    var.set(options[0])
                self.inputs[key] = var

            elif field_type == "multiselect":
                options = field.get("options", [])
                listbox = tk.Listbox(self, selectmode="multiple", width=37, height=6)
                for opt in options:
                    listbox.insert("end", opt)
                listbox.grid(row=row, column=1, padx=5)
                self.inputs[key] = listbox

            row += 1

        # OK button
        tk.Button(self, text="OK", width=15, command=self.submit)\
            .grid(row=row, column=0, columnspan=2, pady=15)

        self.grab_set()  # Make dialog modal

    def submit(self):
        """Collect values from all widgets and store in result."""
        for key, widget in self.inputs.items():

            # For normal text or dropdown fields
            if isinstance(widget, (tk.StringVar)):
                self.result[key] = widget.get()

            # For Listbox multi-select
            elif isinstance(widget, tk.Listbox):
                selections = [widget.get(i) for i in widget.curselection()]
                self.result[key] = selections

        self.destroy()
