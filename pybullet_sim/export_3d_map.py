import plotly.graph_objects as go
import numpy as np

def get_sphere_mesh(x, y, z, r, num_points=12):
    phi = np.linspace(0, np.pi, num_points)
    theta = np.linspace(0, 2 * np.pi, num_points)
    phi_grid, theta_grid = np.meshgrid(phi, theta)
    x_grid = r * np.sin(phi_grid) * np.cos(theta_grid) + x
    y_grid = r * np.sin(phi_grid) * np.sin(theta_grid) + y
    z_grid = r * np.cos(phi_grid) + z
    return x_grid, y_grid, z_grid

def get_cylinder_mesh(x, y, r, h, num_points=8):
    theta = np.linspace(0, 2*np.pi, num_points)
    z = np.linspace(0, h, 2)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = r * np.cos(theta_grid) + x
    y_grid = r * np.sin(theta_grid) + y
    return x_grid, y_grid, z_grid

def export_3d_html_map(trees, drone_path, goal_pos, output_file="pybullet_sim/flight_gym_realistic_map.html"):
    """
    Exports an interactive 3D map of the PyBullet simulation using Plotly.
    """
    fig = go.Figure()

    # Draw obstacles as true 3D surfaces
    for tree in trees:
        x, y, r, h = tree['x'], tree['y'], tree['radius'], tree['height']
        obs_type = tree.get('type', 'cylinder')
        
        if obs_type == 'sphere':
            z = tree.get('z', 1.0)
            X, Y, Z = get_sphere_mesh(x, y, z, r)
            color = 'deepskyblue'
            name = 'Sphere Obstacle'
        else:
            X, Y, Z = get_cylinder_mesh(x, y, r, h)
            color = 'orangered'
            name = 'Cylinder Obstacle'
        
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            colorscale=[[0, color], [1, color]], 
            showscale=False,
            hoverinfo='skip',
            name=name
        ))

    # Draw drone trajectory in 3D
    path_x = [p[0] for p in drone_path]
    path_y = [p[1] for p in drone_path]
    path_z = [p[2] for p in drone_path]
    fig.add_trace(go.Scatter3d(
        x=path_x, y=path_y, z=path_z,
        mode='lines+markers',
        name='UAV Trajectory',
        hoverinfo='skip',
        line=dict(color='blue', width=4),
        marker=dict(size=3, color='blue')
    ))

    # Draw Start
    fig.add_trace(go.Scatter3d(
        x=[path_x[0]], y=[path_y[0]], z=[path_z[0]],
        mode='markers', name='Start', hoverinfo='skip',
        marker=dict(color='green', size=8, symbol='diamond')
    ))

    # Draw Goal
    fig.add_trace(go.Scatter3d(
        x=[goal_pos[0]], y=[goal_pos[1]], z=[goal_pos[2]],
        mode='markers', name='Goal', hoverinfo='skip',
        marker=dict(color='red', size=8, symbol='cross')
    ))

    # Formatting
    fig.update_layout(
        title="3D Interactive Map of PyBullet Forest & CF2X Trajectory",
        scene=dict(
            xaxis_title="X (Right) [m]",
            yaxis_title="Y (Forward) [m]",
            zaxis_title="Z (Up) [m]",
            aspectmode='data', # Keeps aspect ratio 1:1:1
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=1.0) # Set a nice default angled 3D view
            )
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    fig.write_html(output_file)
    print(f"Interactive 3D HTML map saved to: {output_file}")


def export_interactive_html_map(mppi_history, drone_path, goal_pos, trees, output_file="pybullet_sim/flight_gym_jax_interactive.html"):
    """
    Exports an interactive 3D map with a timeline slider for the PyBullet simulation.
    """
    data = []

    path_x = [p[0] for p in drone_path]
    path_y = [p[1] for p in drone_path]
    path_z = [p[2] for p in drone_path]

    # 0. Full Trajectory
    data.append(go.Scatter3d(
        x=path_x, y=path_y, z=path_z,
        mode='lines', name='Full UAV Trajectory',
        line=dict(color='blue', width=2), opacity=0.5
    ))

    # 1. Start Position
    data.append(go.Scatter3d(
        x=[path_x[0]], y=[path_y[0]], z=[path_z[0]],
        mode='markers', name='Start',
        marker=dict(color='green', size=6, symbol='diamond')
    ))

    # 2. Goal Position
    data.append(go.Scatter3d(
        x=[goal_pos[0]], y=[goal_pos[1]], z=[goal_pos[2]],
        mode='markers', name='Goal',
        marker=dict(color='red', size=6, symbol='cross')
    ))

    # 3. Dynamic Voxel Map
    data.append(go.Scatter3d(
        x=[], y=[], z=[],
        mode='markers', name='Voxel Map',
        marker=dict(size=2, color='black', opacity=0.8)
    ))

    # 4. UAV Current Position
    data.append(go.Scatter3d(
        x=[], y=[], z=[],
        mode='markers', name='UAV Current Pos',
        marker=dict(color='orange', size=6)
    ))

    # 5. Best PA-MPPI Path
    data.append(go.Scatter3d(
        x=[], y=[], z=[],
        mode='lines+markers', name='Best PA-MPPI Path',
        line=dict(color='red', width=4),
        marker=dict(size=3, color='red')
    ))

    # 6. PA-MPPI Candidates
    data.append(go.Scatter3d(
        x=[], y=[], z=[], mode='lines', name='PA-MPPI Candidates',
        line=dict(color='purple', width=1.5), opacity=0.15
    ))

    # 7+. PyBullet Obstacles
    for tree in trees:
        x, y, r, h = tree['x'], tree['y'], tree['radius'], tree['height']
        obs_type = tree.get('type', 'cylinder')
        if obs_type == 'sphere':
            z = tree.get('z', 1.0)
            X, Y, Z = get_sphere_mesh(x, y, z, r)
            color = 'deepskyblue'
            name = 'Sphere Obstacle'
        else:
            X, Y, Z = get_cylinder_mesh(x, y, r, h)
            color = 'orangered'
            name = 'Cylinder Obstacle'
        
        data.append(go.Surface(
            x=X, y=Y, z=Z,
            colorscale=[[0, color], [1, color]], 
            showscale=False, hoverinfo='skip', name=name
        ))

    steps = []
    for step in mppi_history:
        f_idx = step['frame_idx']
        p_w = step['p_w']
        best_p = step['best_path']
        cands = step['candidate_paths']
        step_pts = np.array(step['known_points'])
        
        if len(step_pts) > 0:
            vox_x, vox_y, vox_z = step_pts[:, 0].tolist(), step_pts[:, 1].tolist(), step_pts[:, 2].tolist()
        else:
            vox_x, vox_y, vox_z = [], [], []
            
        cand_x, cand_y, cand_z = [], [], []
        for p in cands:
            cand_x.extend(np.round(p[:, 0], 2).tolist() + [None])
            cand_y.extend(np.round(p[:, 1], 2).tolist() + [None])
            cand_z.extend(np.round(p[:, 2], 2).tolist() + [None])
            
        best_x = np.round(best_p[:, 0], 2).tolist()
        best_y = np.round(best_p[:, 1], 2).tolist()
        best_z = np.round(best_p[:, 2], 2).tolist()
        
        slider_step = dict(
            args=[
                {
                    'x': [vox_x, [float(np.round(p_w[0], 2))], best_x, cand_x],
                    'y': [vox_y, [float(np.round(p_w[1], 2))], best_y, cand_y],
                    'z': [vox_z, [float(np.round(p_w[2], 2))], best_z, cand_z]
                },
                [3, 4, 5, 6]
            ],
            label=f"Step {f_idx}",
            method='restyle'
        )
        steps.append(slider_step)
        
    sliders = [dict(
        active=0, currentvalue=dict(prefix="Simulation Step: ", font=dict(size=16), visible=True),
        pad=dict(t=50), steps=steps
    )]
    
    layout = go.Layout(
        title=dict(text="PyBullet PA-MPPI Trajectory Planning Evolution", font=dict(size=20)),
        scene=dict(
            xaxis=dict(title='X (Right)'),
            yaxis=dict(title='Y (Forward)'),
            zaxis=dict(title='Z (Up)'),
            aspectmode='data'
        ),
        sliders=sliders, template="plotly_white", margin=dict(l=0, r=0, b=0, t=50)
    )
    
    fig = go.Figure(data=data, layout=layout)
    fig.write_html(output_file)
    print(f"Interactive timeline HTML map saved to {output_file}")

