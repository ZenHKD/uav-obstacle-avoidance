import os
import sys
import numpy as np
import cv2
import pybullet as p
import importlib.util
from scipy.spatial.transform import Rotation
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

def get_camera_image(pos, rpy, client_id=0):
    yaw = rpy[2]
    rot = Rotation.from_euler('xyz', [0.0, 0.0, yaw])
    rot_mat = rot.as_matrix()
    forward = rot_mat[:, 1]
    up = np.array([0.0, 0.0, 1.0])
    eye_pos = pos + forward * 0.15
    target = eye_pos + forward * 1.0
    view_matrix = p.computeViewMatrix(cameraEyePosition=eye_pos, cameraTargetPosition=target, cameraUpVector=up, physicsClientId=client_id)
    projection_matrix = p.computeProjectionMatrixFOV(69.0, 1.0, 0.1, 10.0, physicsClientId=client_id)
    
    # Randomize Lighting (Direction, Color, Ambient)
    light_dir = [np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(0.2, 1.5)]
    light_color = [np.random.uniform(0.6, 1.0), np.random.uniform(0.6, 1.0), np.random.uniform(0.6, 1.0)]
    ambient = np.random.uniform(0.3, 0.8)
    diffuse = np.random.uniform(0.5, 1.0)
    
    width, height, rgb, depth, seg = p.getCameraImage(
        224, 224, 
        viewMatrix=view_matrix, 
        projectionMatrix=projection_matrix, 
        lightDirection=light_dir,
        lightColor=light_color,
        lightDistance=10.0,
        lightAmbientCoeff=ambient,
        lightDiffuseCoeff=diffuse,
        shadow=1,
        renderer=p.ER_BULLET_HARDWARE_OPENGL, 
        physicsClientId=client_id
    )
    rgb_np = np.reshape(rgb, (224, 224, 4)).astype(np.uint8)[:, :, :3]
    depth_buffer = np.reshape(depth, [224, 224])
    near, far = 0.1, 10.0
    linear_depth = far * near / (far - (far - near) * depth_buffer)
    return rgb_np, linear_depth

def spawn_random_environment(client_id):
    p.resetSimulation(physicsClientId=client_id)
    p.setGravity(0, 0, -9.8, physicsClientId=client_id)
    
    import pybullet_data
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
    floor_id = p.loadURDF("plane.urdf", [0, 0, 0], physicsClientId=client_id)
    
    obstacle_ids = [floor_id]
    
    # 1. Vertical Cylinders
    num_cylinders = np.random.randint(10, 20)
    for _ in range(num_cylinders):
        r = np.random.uniform(0.1, 0.4)
        x = np.random.uniform(-4.0, 4.0)
        y = np.random.uniform(-2.0, 15.0)
        col_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=5.0)
        color = [np.random.rand(), np.random.rand(), np.random.rand(), 1.0]
        vis_id = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=5.0, rgbaColor=color)
        uid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vis_id, basePosition=[x, y, 2.5], physicsClientId=client_id)
        obstacle_ids.append(uid)
        
    # 2. Floating Spheres
    num_spheres = np.random.randint(5, 15)
    for _ in range(num_spheres):
        r = np.random.uniform(0.15, 0.5)
        x = np.random.uniform(-4.0, 4.0)
        y = np.random.uniform(-2.0, 15.0)
        z = np.random.uniform(0.5, 3.5)
        col_id = p.createCollisionShape(p.GEOM_SPHERE, radius=r)
        color = [np.random.rand(), np.random.rand(), np.random.rand(), 1.0]
        vis_id = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=color)
        uid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vis_id, basePosition=[x, y, z], physicsClientId=client_id)
        obstacle_ids.append(uid)

    # 3. Random Rotated Boxes (Walls, Beams, Obstruction Panels)
    num_boxes = np.random.randint(10, 20)
    for _ in range(num_boxes):
        hx = np.random.uniform(0.1, 1.0)
        hy = np.random.uniform(0.1, 1.0)
        hz = np.random.uniform(0.1, 1.0)
        x = np.random.uniform(-4.0, 4.0)
        y = np.random.uniform(-2.0, 15.0)
        z = np.random.uniform(0.5, 3.5)
        
        # Ngẫu nhiên xoay hộp các hướng khác nhau
        quat = p.getQuaternionFromEuler([np.random.uniform(-3.14, 3.14), np.random.uniform(-3.14, 3.14), np.random.uniform(-3.14, 3.14)])
        
        col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz])
        color = [np.random.rand(), np.random.rand(), np.random.rand(), 1.0]
        vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, hz], rgbaColor=color)
        uid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vis_id, basePosition=[x, y, z], baseOrientation=quat, physicsClientId=client_id)
        obstacle_ids.append(uid)
        
    return obstacle_ids

def main():
    out_dir = "dataset_pybullet"
    os.makedirs(os.path.join(out_dir, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "depth"), exist_ok=True)
    
    PYB_CLIENT = p.connect(p.DIRECT)
    egl_spec = importlib.util.find_spec('eglRenderer')
    if egl_spec and egl_spec.origin:
        p.loadPlugin(egl_spec.origin, "_eglRendererPlugin", physicsClientId=PYB_CLIENT)

    labels_csv = open(os.path.join(out_dir, "labels.csv"), "w")
    labels_csv.write("frame,x,y,z,qx,qy,qz,qw,dist_gt\n")

    print("Generating PyBullet Domain Randomization Data...")
    
    num_frames = 20000
    frame_idx = 0
    
    dummy_camera_id = -1
    obstacle_ids = []
    
    pbar = tqdm(total=num_frames)
    while frame_idx < num_frames:
        if frame_idx % 50 == 0:
            obstacle_ids = spawn_random_environment(PYB_CLIENT)
            # Tạo một Dummy Sphere bé xíu (0.01m) đại diện cho Camera để tính khoảng cách vật lý
            dummy_col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.01)
            dummy_camera_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=dummy_col, basePosition=[0,0,0], physicsClientId=PYB_CLIENT)
            
        rx = np.random.uniform(-3.0, 3.0)
        ry = np.random.uniform(-2.0, 15.0)
        rz = np.random.uniform(0.3, 3.5)
        
        rpitch = np.random.uniform(-0.4, 0.4)
        rroll = np.random.uniform(-0.2, 0.2)
        ryaw = np.random.uniform(-3.14, 3.14) # Look in all directions!
        
        pos = np.array([rx, ry, rz])
        rpy = np.array([rroll, rpitch, ryaw])
        
        rgb, depth = get_camera_image(pos, rpy, PYB_CLIENT)
        
        # Đặt Dummy Sphere vào vị trí Camera và dùng Physics Engine của PyBullet để đo khoảng cách chính xác tới TẤT CẢ các vật thể (Cylinder, Box, Sphere, Floor)
        p.resetBasePositionAndOrientation(dummy_camera_id, pos, [0,0,0,1], physicsClientId=PYB_CLIENT)
        
        min_dist = float('inf')
        for obs_id in obstacle_ids:
            pts = p.getClosestPoints(bodyA=dummy_camera_id, bodyB=obs_id, distance=10.0, physicsClientId=PYB_CLIENT)
            if pts:
                obs_dists = [pt[8] for pt in pts]
                closest = min(obs_dists)
                if closest < min_dist:
                    min_dist = closest
                    
        if min_dist != float('inf'):
            dist_gt = min_dist + 0.01 # Cộng lại bán kính 0.01 để lấy khoảng cách từ TÂM camera
        else:
            dist_gt = 10.0
            
        dist_gt = np.clip(dist_gt, 0.0, 10.0)
        
        # Don't save frames if the drone is generated INSIDE an obstacle (dist_gt < 0.1m)
        if dist_gt < 0.1:
            continue
            
        rgb_path = os.path.join(out_dir, "rgb", f"{frame_idx:04d}.jpg")
        cv2.imwrite(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        
        depth_path = os.path.join(out_dir, "depth", f"{frame_idx:04d}.npy")
        np.save(depth_path, depth.astype(np.float32))
        
        quat = p.getQuaternionFromEuler(rpy)
        labels_csv.write(f"{frame_idx},{rx},{ry},{rz},{quat[0]},{quat[1]},{quat[2]},{quat[3]},{dist_gt:.4f}\n")
        
        frame_idx += 1
        pbar.update(1)

    labels_csv.close()
    p.disconnect(PYB_CLIENT)
    print(f"\nDone! Generated {num_frames} randomized frames in {out_dir}/")

if __name__ == "__main__":
    main()
