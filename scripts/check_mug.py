from pxr import Usd, UsdGeom

stage = Usd.Stage.Open('C:/Users/user/Desktop/isaacProject/SM_Mug_D1.usd')
prim = stage.GetDefaultPrim()
print('Default Prim:', prim.GetPath())
bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
bbox = bbox_cache.ComputeWorldBound(prim)
box = bbox.ComputeAlignedBox()
print('Min:', box.GetMin())
print('Max:', box.GetMax())
print('Size:', box.GetSize())

