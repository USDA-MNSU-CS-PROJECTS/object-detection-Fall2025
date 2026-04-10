# App configuration constants (model names, noise profiles).
import os
import sys

_third = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "third_party"))
if _third not in sys.path:
    sys.path.insert(0, _third)
