__version__ = '0.0.1'
__author__ = 'gzh'
__email__ = 'guozhenhao17@mails.ucas.ac.cn'

import sys
from . import model as model
from . import preprocess as pp
from . import tools as tl

sys.modules.update({f"{__name__}.{m}": globals()[m] for m in ["model", "pp", "tl"]})

__all__ = ["model", "pp", "tl"]
