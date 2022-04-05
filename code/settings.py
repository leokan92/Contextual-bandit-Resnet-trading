import os
from pathlib import Path


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(Path(BASE_DIR).parent, 'code', 'data')
RESULTS_DIR = os.path.join(Path(BASE_DIR).parent,'data')
SAVE_PARAMS = os.path.join(Path(BASE_DIR).parent, 'code', 'params')
