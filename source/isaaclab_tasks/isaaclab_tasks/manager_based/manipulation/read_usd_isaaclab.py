import sys
import os

# Import IsaacLab to set up paths
try:
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher({"headless": True})
    simulation_app = app_launcher.app
except Exception as e:
    print(f"IsaacLab init failed: {e}")

try:
    from pxr import Usd
    usd_path = r"C:\Users\user\Desktop\IsaacLab\openarm_isaac_lab\source\openarm\openarm\tasks\manager_based\openarm_manipulation\usds\openarm_bimanual\openarm_bimanual.usd"
    print(f"Opening {usd_path}")
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        print("Failed to open stage")
        sys.exit(1)

    print("--- USD Hierarchy ---")
    for p in stage.Traverse():
        print(p.GetPath().pathString)
except Exception as e:
    print(f"Error: {e}")
