
"""
manager.py

Central override storage for the Validator GUI + backend system.

The override manager:

- Stores user override values
- Provides get/set/has access
- Can be cleared entirely
- Used by both main.py and app.py
"""

class OverrideManager:

    def __init__(self):
        self.overrides = {}

    def set(self, key, value):
        """Store a manual override parameter."""
        self.overrides[key] = value

    def get(self, key, default=None):
        """Retrieve override parameter if exists."""
        return self.overrides.get(key, default)

    def has(self, key):
        """Check if override for a given key exists."""
        return key in self.overrides

    def clear(self):
        """Remove all override values."""
        self.overrides = {}

    def as_dict(self):
        """Return overrides as plain dictionary."""
        return dict(self.overrides)
