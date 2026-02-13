#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from api.validation import sanitize_input

dangerous = "<script>alert('xss')</script>"
sanitized = sanitize_input(dangerous)
print("Original:", dangerous)
print("Sanitized:", sanitized)
print("Has <:", '<' in sanitized)
print("Has >:", '>' in sanitized)
print("Has <:", '<' in sanitized)
print("Has >:", '>' in sanitized)
print("Has ":", '"' in sanitized)
print("Has &#x27;:", '&#x27;' in sanitized)
