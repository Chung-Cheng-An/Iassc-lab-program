"""
view_pointcloud_3d.py
=====================
專用 3D 點雲檢視器：
  1. 自動讀取 openarm/cup_pointcloud.ply 點雲檔。
  2. 生成多角度 3D 透視圖照片 head_cam_pointcloud_3d.png (可直接雙擊開照片看！)。
  3. 啟動互動式 3D 彈窗視窗，可以用滑鼠 360 度旋轉、縮放與拖曳 3D 杯子點雲結構！

Run:
    .\\isaaclab.bat -p scripts\\view_pointcloud_3d.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

PLY_PATH = r"C:\Users\user\Desktop\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\manager_based\manipulation\lift\config\openarm\cup_pointcloud.ply"
OUT_IMG_PATH = r"C:\Users\user\Desktop\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\manager_based\manipulation\lift\config\openarm\head_cam_pointcloud_3d.png"


def load_ply_ascii(filename):
    """手動解析 ASCII PLY 檔案格式。"""
    points = []
    colors = []
    header_ended = False
    
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not header_ended:
                if line == "end_header":
                    header_ended = True
                continue
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 6:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                r, g, b = int(parts[3]), int(parts[4]), int(parts[5])
                points.append([x, y, z])
                colors.append([r / 255.0, g / 255.0, b / 255.0])
                
    return np.array(points), np.array(colors)


def main():
    if not os.path.exists(PLY_PATH):
        print(f"[錯誤] 找不到點雲檔案: {PLY_PATH}")
        print("請先執行 .\\isaaclab.bat -p scripts\\generate_cup_pointcloud.py 生成點雲！")
        return

    print(f"[INFO] 載入點雲檔案中: {PLY_PATH}")
    points, colors = load_ply_ascii(PLY_PATH)
    print(f"[INFO] 成功載入 {len(points)} 個 3D 空間點！")

    # 1. 自動繪製多角度 3D 透視圖 PNG 照片 (可以直接打開 PNG 照片看！)
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 3D 散點圖繪製
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=2, alpha=0.8)
    
    ax.set_xlabel("X (Robot Forward) [m]")
    ax.set_ylabel("Y (Robot Left) [m]")
    ax.set_zlabel("Z (Height) [m]")
    ax.set_title("Cup 3D Point Cloud Reconstruction (Robot Base Frame)", fontsize=14)
    
    # 視角最佳調整
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    plt.savefig(OUT_IMG_PATH, dpi=150)
    print(f"\n[SUCCESS] 已生成多角度 3D 透視圖照片: {OUT_IMG_PATH}")
    print("您現在可以直接打開 PNG 照片檢視點雲！\n")

    # 2. 嘗試使用 Open3D 彈出 360 度互動視窗 (若有安裝 open3d)
    try:
        import open3d as o3d
        print("[INFO] 啟動 Open3D 360 度互動檢視視窗 (用滑鼠拖曳旋轉/滾輪縮放)...")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd], window_name="OpenArm - Cup 3D Point Cloud Viewer")
    except ImportError:
        # 若未安裝 open3d，則使用 Matplotlib 的互動彈窗
        print("[INFO] 啟動 Matplotlib 3D 互動視窗 (用滑鼠左鍵旋轉/右鍵縮放)...")
        plt.show()


if __name__ == "__main__":
    main()
