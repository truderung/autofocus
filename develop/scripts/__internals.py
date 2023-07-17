import tempfile
from pathlib import Path
import re

TEMP = tempfile.gettempdir()
WORK_PATH = Path(__file__).parent.parent
SRC_PATH = Path(WORK_PATH).parent
OUT_PATH = Path(WORK_PATH, "out")
Path.mkdir(OUT_PATH, exist_ok=True)
print(WORK_PATH)
exit(0)
# This just as long as auto_focusing is not a package ################
import sys
sys.path.append(str(WORK_PATH))
# ######################################

def __get_out_dir_nr():
  pathlist = OUT_PATH.glob('exp*')
  exp_nr = 0
  for path in pathlist:
    path_in_str = str(path)
    exp_nr = max(exp_nr, int(re.search(r"(\d+)", path_in_str).group(1)))
  return exp_nr


def get_out_dir(exp_nr, create_if_not_exist = False):
  out = Path(OUT_PATH, f"exp{exp_nr}")
  if create_if_not_exist:
    Path.mkdir(out)
  return out


def get_new_out_dir():
  exp_nr = __get_out_dir_nr()
  out = Path(OUT_PATH, f"exp{exp_nr+1}")
  Path.mkdir(out)
  return out


def get_last_out_dir():
  exp_nr = __get_out_dir_nr()

  if exp_nr == 0:
    return None
  else:
    return Path(OUT_PATH, f"exp{exp_nr}")