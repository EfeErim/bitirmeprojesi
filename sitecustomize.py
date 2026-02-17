"""Site customization to fix Python 3.13 warnings filter issue."""

import warnings

# Completely replace the warnings filter management to avoid the list.remove() error
# Save original filters list
if hasattr(warnings, 'filters'):
    _original_filters = warnings.filters.copy()
else:
    _original_filters = []

# Create a custom filter management system
class FilterManager:
    def __init__(self):
        self.filters = _original_filters.copy()
    
    def add_filter(self, action, message='', category=Warning, lineno=0, append=False):
        """Add a filter, handling duplicates gracefully."""
        try:
            # Try to remove existing filter with same signature to avoid duplicates
            for i, (a, m, c, l) in enumerate(self.filters):
                if (a == action and m == message and c == category and l == lineno):
                    del self.filters[i]
                    break
        except Exception:
            pass
        
        # Add the new filter
        self.filters.append((action, message, category, lineno))
    
    def get_filters(self):
        return self.filters

# Create global filter manager
_filter_manager = FilterManager()

# Replace warnings functions
def _patched_add_filter(*args, **kwargs):
    """Patched version that uses our filter manager."""
    # Handle both positional and keyword arguments flexibly
    if args:
        action = args[0]
        message = args[1] if len(args) > 1 else kwargs.get('message', '')
        category = args[2] if len(args) > 2 else kwargs.get('category', Warning)
        lineno = args[3] if len(args) > 3 else kwargs.get('lineno', 0)
        append = args[4] if len(args) > 4 else kwargs.get('append', False)
    else:
        action = kwargs.get('action')
        message = kwargs.get('message', '')
        category = kwargs.get('category', Warning)
        lineno = kwargs.get('lineno', 0)
        append = kwargs.get('append', False)
    
    _filter_manager.add_filter(action, message, category, lineno, append)

def _patched_simplefilter(action, category=Warning, lineno=0, append=False):
    """Patched version that uses our filter manager."""
    _filter_manager.add_filter(action, '', category, lineno, append)

def _patched_reset_filters():
    """Reset filters to original state."""
    _filter_manager.filters = _original_filters.copy()

# Apply patches
if hasattr(warnings, '_add_filter'):
    warnings._add_filter = _patched_add_filter
warnings.simplefilter = _patched_simplefilter

# Also provide a way to get the current filters (for internal use)
def get_current_filters():
    return _filter_manager.get_filters()

# Reset filters to known good state
_patched_reset_filters()
