from pxr import Usd
import os

usd_path = "openarm_isaac_lab/source/openarm/openarm/tasks/manager_based/openarm_manipulation/usds/openarm_bimanual/openarm_bimanual.usd"
print(f"Loading USD from {usd_path}")
stage = Usd.Stage.Open(usd_path)
for prim in stage.Traverse():
    print(prim.GetPath())
