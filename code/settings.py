import os
from pathlib import Path


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(Path(BASE_DIR).parent,'data')
RESULTS_DIR = os.path.join(Path(BASE_DIR).parent, 'code','results')
SAVE_PARAMS = os.path.join(Path(BASE_DIR).parent, 'code', 'params')
