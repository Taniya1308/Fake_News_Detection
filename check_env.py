import sys
print("Python:", sys.version)
try:
    import sklearn; print("scikit-learn:", sklearn.__version__)
except ImportError as e: print("MISSING:", e)
try:
    import nltk; print("nltk:", nltk.__version__)
except ImportError as e: print("MISSING:", e)
try:
    import pandas; print("pandas:", pandas.__version__)
except ImportError as e: print("MISSING:", e)
try:
    import numpy; print("numpy:", numpy.__version__)
except ImportError as e: print("MISSING:", e)
try:
    import matplotlib; print("matplotlib:", matplotlib.__version__)
except ImportError as e: print("MISSING:", e)
print("All checks done.")
