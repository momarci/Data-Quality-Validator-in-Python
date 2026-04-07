
"""
registry.py

Maps error messages or step failures to override dialog definitions.

The Validator backend can request a dialog for override by passing a
dictionary describing input fields. This registry provides reusable
definitions for common override scenarios.
"""


class OverrideRegistry:

    def __init__(self):
        # This dictionary maps keys (override scenario names)
        # to dialog templates.
        self.templates = {
            "choose_date_column": {
                "title": "Select Date Column",
                "fields": {
                    "date_column": {
                        "type": "dropdown",
                        "label": "Multiple date-like columns found.\nSelect the correct date column:",
                        "options": []  # filled dynamically
                    }
                }
            },

            "frequency_override": {
                "title": "Frequency Required",
                "fields": {
                    "frequency": {
                        "type": "text",
                        "label": "Cannot infer frequency.\nEnter frequency (D/M/Q/W/etc.):"
                    }
                }
            },

            "stl_period_override": {
                "title": "STL Seasonal Period",
                "fields": {
                    "seasonal_period": {
                        "type": "text",
                        "label": "Enter STL seasonal period (e.g., 7, 12, 24):"
                    }
                }
            },

            "missing_date_strategy": {
                "title": "Missing Date Strategy",
                "fields": {
                    "missing_dates_action": {
                        "type": "dropdown",
                        "label": "Missing dates detected.\nSelect handling method:",
                        "options": ["ignore", "forward_fill", "interpolate", "drop"]
                    }
                }
            },

            "choose_entity_column": {
                "title": "Select Entity Column",
                "fields": {
                    "entity_column": {
                        "type": "dropdown",
                        "label": "Select the entity identifier column:",
                        "options": []
                    }
                }
            },

            "select_value_column": {
                "title": "Select Value Column",
                "fields": {
                    "value_column": {
                        "type": "dropdown",
                        "label": "Choose the numeric value column for time-series analysis:",
                        "options": []
                    }
                }
            },

            "select_y_x_columns": {
                "title": "Regression Column Selection",
                "fields": {
                    "y_column": {
                        "type": "dropdown",
                        "label": "Select dependent variable (Y):",
                        "options": []
                    },
                    "x_columns": {
                        "type": "multiselect",
                        "label": "Select independent variables (X):",
                        "options": []
                    }
                }
            }
        }

    def get(self, key):
        """Return an override template by key."""
        return self.templates.get(key)
