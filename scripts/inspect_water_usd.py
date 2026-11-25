import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom

stage = Usd.Stage.Open("C:/Users/user/Desktop/isaacProject/SM_Mug_Water.usd")
with open("water_prims.txt", "w", encoding="utf-8") as f:
    f.write("=== All Prims in SM_Mug_Water.usd ===\n")
    for prim in stage.Traverse():
        f.write(f"Path: {prim.GetPath()}, Type: {prim.GetTypeName()}\n")

stage_hollow = Usd.Stage.Open("C:/Users/user/Desktop/isaacProject/SM_Mug_Hollow.usd")
with open("hollow_prims.txt", "w", encoding="utf-8") as f:
    f.write("=== All Prims in SM_Mug_Hollow.usd ===\n")
    for prim in stage_hollow.Traverse():
        f.write(f"Path: {prim.GetPath()}, Type: {prim.GetTypeName()}\n")

simulation_app.close()
