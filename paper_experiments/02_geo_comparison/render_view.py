import torch
import os
import struct
import numpy as np
import scipy.ndimage  # 用于填补点云空洞，让深度图更实
from plyfile import PlyData

# ==============================================================================
# 🔴 必填配置 (请确认路径)
# ==============================================================================
PLY_PATH = "/home/ubuntu/lyj/Project/GlowGS/output/bicycle/point_cloud/iteration_30000/point_cloud.ply"
SOURCE_PATH = "/home/ubuntu/lyj/Project/GlowGS/data/360_v2/bicycle"
OUTPUT_FILENAME = "ours_view0.npz"

# 渲染参数
SPLAT_SCALE = 5.0    # 稍微放大一点点，填补缝隙
OPACITY_THRESHOLD = 0.1
# ==============================================================================

def qvec2rotmat(qvec):
    qvec = np.array(qvec, dtype=np.float32)
    w, x, y, z = qvec
    R = np.zeros((3, 3), dtype=np.float32)
    R[0, 0] = 1 - 2 * (y**2 + z**2)
    R[0, 1] = 2 * (x*y - w*z)
    R[0, 2] = 2 * (x*z + w*y)
    R[1, 0] = 2 * (x*y + w*z)
    R[1, 1] = 1 - 2 * (x**2 + z**2)
    R[1, 2] = 2 * (y*z - w*x)
    R[2, 0] = 2 * (x*z - w*y)
    R[2, 1] = 2 * (y*z + w*x)
    R[2, 2] = 1 - 2 * (x**2 + y**2)
    return R

def read_cameras_binary(path_to_model_file):
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_cameras):
            camera_properties = struct.unpack("<iiQQ", fid.read(24))
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]
            if model_id == 0: 
                params = struct.unpack("<3d", fid.read(24))
                cameras[camera_id] = (width, height, params[0], params[0], params[1], params[2])
            elif model_id == 1: 
                params = struct.unpack("<4d", fid.read(32))
                cameras[camera_id] = (width, height, params[0], params[1], params[2], params[3])
            else:
                params = struct.unpack("<4d", fid.read(32))
                cameras[camera_id] = (width, height, params[0], params[1], params[2], params[3])
    return cameras

def read_images_binary(path_to_model_file):
    images = []
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_reg_images):
            binary_image_properties = struct.unpack("<idddddddi", fid.read(64))
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = struct.unpack("<c", fid.read(1))[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = struct.unpack("<c", fid.read(1))[0]
            num_points2D = struct.unpack("<Q", fid.read(8))[0]
            fid.read(num_points2D * 24)
            images.append({"id": image_id, "qvec": qvec, "tvec": tvec, "camera_id": camera_id, "name": image_name})
    return images

# --- 加载 PLY 并计算法线 (GPU加速) ---
def build_rotation(r):
    norm = torch.sqrt(r[:,0]**2 + r[:,1]**2 + r[:,2]**2 + r[:,3]**2)
    q = r / norm[:, None]
    R = torch.zeros((r.shape[0], 3, 3), device="cuda")
    r, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def load_ply_data(path):
    print(f"Loading PLY: {path}")
    plydata = PlyData.read(path)
    v = plydata['vertex']
    
    xyz = np.stack((v['x'], v['y'], v['z']), axis=-1)
    op = np.asarray(v['opacity'])
    opacity = 1.0 / (1.0 + np.exp(-op))
    
    # 颜色
    try:
        rgb = np.stack((v['f_dc_0'], v['f_dc_1'], v['f_dc_2']), axis=-1)
        rgb = 0.282 * rgb + 0.5
        rgb = np.clip(rgb, 0, 1)
    except:
        rgb = np.ones_like(xyz) * 0.5

    # 法线计算 (基于旋转矩阵的最短轴)
    s_names = [n for n in v.data.dtype.names if n.startswith('scale')]
    r_names = [n for n in v.data.dtype.names if n.startswith('rot')]
    scales = np.stack([v[n] for n in s_names], axis=-1)
    rots = np.stack([v[n] for n in r_names], axis=-1)
    
    scales_t = torch.tensor(scales, device="cuda", dtype=torch.float32)
    rots_t = torch.tensor(rots, device="cuda", dtype=torch.float32)
    
    R = build_rotation(rots_t)
    min_scale_idx = torch.argmin(scales_t, dim=1, keepdim=True)
    basis_vector = torch.zeros_like(scales_t)
    basis_vector.scatter_(1, min_scale_idx, 1.0)
    normals = torch.bmm(R, basis_vector.unsqueeze(-1)).squeeze(-1)
    normals = torch.nn.functional.normalize(normals, dim=1)
    
    radius = torch.exp(torch.max(scales_t, dim=1).values)

    return {
        'xyz': torch.tensor(xyz, device="cuda", dtype=torch.float32),
        'rgb': torch.tensor(rgb, device="cuda", dtype=torch.float32),
        'radius': radius,
        'opacity': torch.tensor(opacity, device="cuda", dtype=torch.float32),
        'normal': normals
    }

# --- 🔥 关键修改：渲染逻辑 (Fixes Black Depth) ---
def robust_render(data, R, T, fx, fy, cx, cy, W, H):
    xyz = data['xyz']
    rgb = data['rgb']
    radius = data['radius'] * SPLAT_SCALE
    opacity = data['opacity']
    normals = data['normal']
    
    # World -> Camera
    R_torch = torch.tensor(R, device="cuda", dtype=torch.float32)
    T_torch = torch.tensor(T, device="cuda", dtype=torch.float32)
    xyz_cam = (xyz @ R_torch.T) + T_torch
    normals_cam = normals @ R_torch.T
    
    # Z-Culling
    mask = xyz_cam[:, 2] > 0.1
    if not mask.any(): return None, None, None
    
    xyz_cam, rgb, radius, opacity, normals_cam = \
        xyz_cam[mask], rgb[mask], radius[mask], opacity[mask], normals_cam[mask]

    # Projection
    z = xyz_cam[:, 2]
    x = xyz_cam[:, 0]
    y = xyz_cam[:, 1]
    u = (x * fx / z) + cx
    v = (y * fy / z) + cy
    
    # Screen Culling
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z, rgb, op, normals_cam = \
        u[valid].long(), v[valid].long(), z[valid], rgb[valid], opacity[valid], normals_cam[valid]
    
    # Sort (Far -> Near)
    sort_idx = torch.argsort(z, descending=True)
    u, v, z, rgb, op, normals_cam = \
        u[sort_idx], v[sort_idx], z[sort_idx], rgb[sort_idx], op[sort_idx], normals_cam[sort_idx]
    
    # 过滤透明点
    solid_mask = op > OPACITY_THRESHOLD
    u, v, z, rgb, normals_cam = \
        u[solid_mask], v[solid_mask], z[solid_mask], rgb[solid_mask], normals_cam[solid_mask]
    
    # --- 缓冲区初始化 ---
    indices = v * W + u
    
    # RGB (Black Background)
    rgb_map = torch.zeros((H, W, 3), device="cuda")
    # Normal (Black Background)
    normal_map = torch.zeros((H, W, 3), device="cuda")
    # Depth (初始化为 NaN 或 Inf，这至关重要！)
    depth_map = torch.full((H, W), np.inf, device="cuda") 

    # 写入 Buffer (覆盖模式)
    rgb_map.view(-1, 3).index_copy_(0, indices, rgb)
    normal_map.view(-1, 3).index_copy_(0, indices, normals_cam)
    depth_map.view(-1).index_copy_(0, indices, z) # Inf 会被有效的 z 值覆盖
    
    # 转回 CPU
    rgb_npy = rgb_map.cpu().numpy()
    norm_npy = normal_map.cpu().numpy()
    depth_npy = depth_map.cpu().numpy()
    
    # --- 🚀 后处理：孔洞填充 (Hole Filling) ---
    print("应用形态学填充，让物体更实...")
    
    # 1. 深度图：将 Inf 设为 NaN，然后用 min filter 扩散前景 (前景值小，背景值大)
    # 实际上由于初始化是 Inf，我们先处理 Inf
    valid_mask = np.isfinite(depth_npy)
    
    # 使用 Grey Erosion (最小值滤波) 来让近处的像素(小值) 膨胀，填补空隙
    # size=3 表示 3x3 邻域
    depth_filled = scipy.ndimage.grey_erosion(depth_npy, size=(3,3))
    
    # 将依然是 Inf 的背景设为 NaN (Matplotlib 能识别并涂黑)
    depth_filled[depth_filled == np.inf] = np.nan

    # 2. RGB 和 Normal 不做填充，保持锐利，或者只对极小的孔洞做闭运算
    # 这里保持原样即可，深度图才是关键
    
    return rgb_npy, depth_filled, norm_npy

def main():
    print(f"--- 增强版渲染脚本 (Safe Mode) ---")
    
    # 读取 Colmap 相机
    sparse_dir = os.path.join(SOURCE_PATH, "sparse", "0")
    cameras = read_cameras_binary(os.path.join(sparse_dir, "cameras.bin"))
    images = read_images_binary(os.path.join(sparse_dir, "images.bin"))
    images.sort(key=lambda x: x["name"])
    target = images[0]
    
    cam = cameras[target["camera_id"]]
    W, H = int(cam[0]), int(cam[1])
    fx, fy, cx, cy = cam[2], cam[3], cam[4], cam[5]
    R = qvec2rotmat(target["qvec"])
    T = target["tvec"]
    
    print(f"相机: {target['name']} ({W}x{H})")
    
    # 读取 PLY
    data = load_ply_data(PLY_PATH)
    
    # 渲染
    print("渲染中...")
    rgb, depth, norm = robust_render(data, R, T, fx, fy, cx, cy, W, H)
    
    if rgb is None:
        print("❌ 渲染失败：没有点投影到画面内。")
        return

    # 保存
    save_path = os.path.join(os.getcwd(), OUTPUT_FILENAME)
    np.savez(save_path, rgb=rgb, depth=depth, norm=norm)
    print(f"✅ 保存成功: {save_path}")

if __name__ == "__main__":
    main()