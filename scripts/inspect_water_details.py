import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom

stage = Usd.Stage.Open("C:/Users/user/Desktop/isaacProject/SM_Mug_Water.usd")
water_prim = stage.GetPrimAtPath("/Root/SM_Mug_D1/Water")
with open("water_details.txt", "w", encoding="utf-8") as f:
    f.write(f"Water prim: {water_prim}\n")
    cyl = UsdGeom.Cylinder(water_prim)
    f.write(f"Radius: {cyl.GetRadiusAttr().Get()}\n")
    f.write(f"Height: {cyl.GetHeightAttr().Get()}\n")
    f.write(f"Axis: {cyl.GetAxisAttr().Get()}\n")
    xf = UsdGeom.Xformable(water_prim)
    f.write(f"Xform ops: {[op.GetName() for op in xf.GetOrderedXformOps()]}\n")
    for op in xf.GetOrderedXformOps():
        f.write(f"Op: {op.GetName()} {op.Get()}\n")

simulation_app.close()