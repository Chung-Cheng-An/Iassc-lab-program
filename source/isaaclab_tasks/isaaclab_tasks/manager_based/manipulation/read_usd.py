import sys
from pxr import Usd

usd_path = r"C:\Users\user\Desktop\IsaacLab\openarm_isaac_lab\source\openarm\openarm\tasks\manager_based\openarm_manipulation\usds\openarm_bimanual\openarm_bimanual.usd"
print(f"Opening {usd_path}")
stage = Usd.Stage.Open(usd_path)
if not stage:
    print("Failed to open stage")
    sys.exit(1)

for p in stage.Traverse():
    path_str = p.GetPath().pathString.lower()
    if 'joint' in path_str or 'ee' in path_str or 'hand' in path_str:
        print(p.GetPath().pathString)
