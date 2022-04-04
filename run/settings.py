import os
from pathlib import Path


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(Path(BASE_DIR).parent, 'run', 'data')
RESULTS_DIR = os.path.join(Path(BASE_DIR).parent, 'run', 'results')
