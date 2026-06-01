from pxr import Usd

usd_path = r"C:\Users\user\Desktop\IsaacLab\openarm_isaac_lab\source\openarm\openarm\tasks\manager_based\openarm_manipulation\usds\openarm_unimanual\openarm_unimanual.usd"

stage = Usd.Stage.Open(usd_path)

if stage is None:
    print("USD open failed:", usd_path)
else:
    for p in stage.Traverse():
        path = str(p.GetPath())
        lower = path.lower()
        if "link" in lower or "hand" in lower or "base" in lower:
            print(path)
