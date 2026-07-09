import pybullet as p
import random
import numpy as np

def spawn_straight_racing_track(client_id=0, seed=123):
    """
    Spawns a straight racing track along the Y-axis with boundary walls and obstacles.
    Goal will be at [0.0, 40.0, 1.0].
    """
    trees = []
    
    # Track parameters
    track_width = 7.0
    wall_thickness = 0.2
    wall_height = 2.5
    
    # Parametric curve: straight line along Y-axis from y=0 to y=40
    num_segments = 25
    t_vals = np.linspace(0, 1.0, num_segments + 1)
    
    for i in range(num_segments):
        t1, t2 = t_vals[i], t_vals[i+1]
        x1 = 0.0
        y1 = 40.0 * t1
        
        x2 = 0.0
        y2 = 40.0 * t2
        
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        angle = np.arctan2(y2 - y1, x2 - x1)
        
        for side in [-1, 1]:  # -1 for left, +1 for right
            wx = cx + side * (track_width / 2.0) * (-np.sin(angle))
            wy = cy + side * (track_width / 2.0) * (np.cos(angle))
            
            col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[length/2, wall_thickness/2, wall_height], physicsClientId=client_id)
            vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[length/2, wall_thickness/2, wall_height], rgbaColor=[0.15, 0.15, 0.15, 1.0], physicsClientId=client_id)
            
            orn = p.getQuaternionFromEuler([0, 0, angle])
            
            p.createMultiBody(baseMass=0, 
                              baseCollisionShapeIndex=col_id,
                              baseVisualShapeIndex=vis_id,
                              basePosition=[wx, wy, wall_height],
                              baseOrientation=orn,
                              physicsClientId=client_id)
                              
    # Spawn obstacles along the track
    random.seed(seed)
    
    # First half: Cylinders (y from 2.0 to 18.0)
    num_cylinders = 9
    y_step = (18.0 - 2.0) / num_cylinders
    for i in range(num_cylinders):
        oy = random.uniform(2.0 + i * y_step, 2.0 + (i + 1) * y_step)
        # Offset laterally within the track
        ox = random.uniform(-track_width/2 + 0.8, track_width/2 - 0.8)
        radius = random.uniform(0.15, 0.35)
        height = 5.0
        z = height / 2.0
        
        col_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height, physicsClientId=client_id)
        vis_id = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=[0.9, 0.3, 0.1, 1], physicsClientId=client_id)
        
        body_id = p.createMultiBody(baseMass=0, 
                                    baseCollisionShapeIndex=col_id,
                                    baseVisualShapeIndex=vis_id,
                                    basePosition=[ox, oy, z],
                                    physicsClientId=client_id)
        trees.append({
            'id': body_id,
            'x': ox,
            'y': oy,
            'z': 0.0,
            'radius': radius,
            'height': height,
            'type': 'cylinder'
        })
        
    # Second half: Floating Spheres (y from 20.0 to 38.0)
    # We spawn spheres in 6 rows: y = 21.0, 24.0, 27.0, 30.0, 33.0, 36.0
    # In each row, we spawn 3 spheres (left, center, right) at different heights (one low, one mid, one high)
    y_rows = [21.0, 24.0, 27.0, 30.0, 33.0, 36.0]
    for oy in y_rows:
        # One low sphere (z: 0.35-0.75m), one mid sphere (z: 0.9-1.25m), one high sphere (z: 1.4-1.8m)
        heights = [random.uniform(0.35, 0.75), random.uniform(0.9, 1.25), random.uniform(1.4, 1.8)]
        random.shuffle(heights)
        
        # X positions: left, center, right
        x_offsets = [
            random.uniform(-track_width/2 + 0.9, -track_width/6),
            random.uniform(-track_width/12, track_width/12),
            random.uniform(track_width/6, track_width/2 - 0.9)
        ]
        
        for ox, z in zip(x_offsets, heights):
            radius = random.uniform(0.25, 0.4)
            height = radius * 2.0
            
            col_id = p.createCollisionShape(p.GEOM_SPHERE, radius=radius, physicsClientId=client_id)
            vis_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=[0.1, 0.6, 0.9, 1.0], physicsClientId=client_id)
            
            body_id = p.createMultiBody(baseMass=0, 
                                        baseCollisionShapeIndex=col_id,
                                        baseVisualShapeIndex=vis_id,
                                        basePosition=[ox, oy, z],
                                        physicsClientId=client_id)
            trees.append({
                'id': body_id,
                'x': ox,
                'y': oy,
                'z': z,
                'radius': radius,
                'height': height,
                'type': 'sphere'
            })
        
    print(f"Spawned straight racing track with {len(trees)} obstacles (cylinders in 1st half, spheres in 2nd half).")
    return trees
