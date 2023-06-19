import os
from pathlib import Path
import sys

print('getcwd:      ', os.getcwd())
print('__file__:    ', __file__)


FILE = Path(__file__).resolve()
print(FILE)
ROOT = FILE.parents[0]
print(ROOT)

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
print(Path.cwd(), ROOT)