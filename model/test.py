
from pathlib import Path
import sys

from torch import t
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

import importlib
unipelt_transformers = importlib.import_module('submodules.2022_Masterthesis_UnifiedPELT.transformers')

get_last_checkpoint = getattr(importlib.import_module('submodules.2022_Masterthesis_UnifiedPELT.transformers.trainer_utils'), 'get_last_checkpoint')
is_main_process = getattr(importlib.import_module('submodules.2022_Masterthesis_UnifiedPELT.transformers.trainer_utils'), 'is_main_process')



print('sucess')