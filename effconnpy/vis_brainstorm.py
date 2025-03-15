import numpy as np
from fury import window, actor
from dipy.io.streamline import load_trk
import os
import time
import vtk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def visualize_trk_with_timeseries(trk_file_path, node_file, edge_file, roi_indices, threshold=0.5, time_series1=None, time_series2=None):
    """
    Load and visualize a .trk file with animated path between nodes and a time series visualization
    
    Parameters
    ----------
    trk_file_path : str
        Path to the .trk file
    node_file : str
        Path to the node coordinates file
    edge_file : str
        Path to the edge/connectivity matrix file
    roi_indices : list of int
        Indices of the ROIs to visualize (rows in the node file)
    threshold : float
        Threshold for the time series signal to activate visualization
    time_series1 : array-like
        First time series data that will modify the cone size during animation
    time_series2 : array-like
        Second time series data that will be plotted alongside the first
    """
    # Check if files exist
    for file_path in [trk_file_path, node_file, edge_file]:
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} does not exist.")
            return
    
    # Create random time series if none provided
    if time_series1 is None:
        time_series1 = np.random.normal(loc=1.0, scale=0.5, size=100)
        time_series1 = np.abs(time_series1)
    if time_series2 is None:
        time_series2 = np.random.normal(loc=1.0, scale=0.5, size=100)
        time_series2 = np.abs(time_series2)
    
    # Load the tractogram with bbox_valid_check=False to bypass validation
    print(f"Loading {trk_file_path}...")
    
    try:
        tractogram = load_trk(trk_file_path, reference="same", bbox_valid_check=False)
        streamlines = tractogram.streamlines
        
        all_points = np.vstack([s for s in streamlines])
        min_bounds = np.min(all_points, axis=0)
        max_bounds = np.max(all_points, axis=0)
        center_of_tracts = (min_bounds + max_bounds) / 2
        print(f"Tract bounds: Min {min_bounds}, Max {max_bounds}, Center {center_of_tracts}")
        
        print(f"Loading {node_file}...")
        
        node_coords = []
        node_sizes = []
        
        with open(node_file, 'r') as f:
            for line in f:
                # Skip comment lines or headers
                if line.startswith('#') or line.strip() == '':
                    continue
                
                # Split the line by whitespace or tabs
                parts = line.strip().split()
                
                # Ensure there are at least 4 columns (x, y, z, size)
                if len(parts) < 4:
                    print(f"Skipping malformed line: {line.strip()}")
                    continue
                
                try:
                    # Extract x, y, z coordinates
                    x = float(parts[0])
                    y = float(parts[1])
                    z = float(parts[2])
                    node_coords.append([x, y, z])
                    
                    # Extract size (4th column) and scale it by 3
                    size = float(parts[3]) * 3  # Scale ROI sizes by 3
                    node_sizes.append(size)
                except ValueError as e:
                    print(f"Skipping line due to parsing error: {line.strip()}")
                    print(f"Error: {e}")
                    continue
        
        # Convert to numpy arrays
        node_coords = np.array(node_coords, dtype=np.float64)
        node_sizes = np.array(node_sizes, dtype=np.float64)
        
        # Calculate the bounding box and center of the connectome
        min_node_bounds = np.min(node_coords, axis=0)
        max_node_bounds = np.max(node_coords, axis=0)
        center_of_nodes = (min_node_bounds + max_node_bounds) / 2
        print(f"Node bounds: Min {min_node_bounds}, Max {max_node_bounds}, Center {center_of_nodes}")
        
        # Calculate the translation needed to align centers
        translation = center_of_tracts - center_of_nodes
        print(f"Translation vector: {translation}")
        
        # Apply translation to node coordinates
        adjusted_node_coords = node_coords + translation
        
        # Ensure ROI indices are valid
        num_nodes = len(adjusted_node_coords)
        roi_indices = [idx for idx in roi_indices if 0 <= idx < num_nodes]
        if not roi_indices:
            print("Error: No valid ROI indices provided.")
            return
        
        # Get coordinates and sizes for the ROIs
        roi_coords = adjusted_node_coords[roi_indices]
        roi_sizes = node_sizes[roi_indices]
        
        # Create a scene
        scene = window.Scene()
        
        # Function to filter streamlines touching the ROIs
        def filter_streamlines(streamlines, roi_coords, roi_sizes):
            filtered_streamlines = []
            for streamline in streamlines:
                for point in streamline:
                    for i, (roi_coord, roi_size) in enumerate(zip(roi_coords, roi_sizes)):
                        distance = np.linalg.norm(point - roi_coord)
                        if distance <= roi_size:
                            filtered_streamlines.append(streamline)
                            break
            return filtered_streamlines
        
        # Add spheres for the ROIs
        for idx, (coord, size) in enumerate(zip(roi_coords, roi_sizes)):
            sphere_source = vtk.vtkSphereSource()
            sphere_source.SetCenter(coord[0], coord[1], coord[2])
            sphere_source.SetRadius(size)
            sphere_source.SetPhiResolution(16)
            sphere_source.SetThetaResolution(16)
            sphere_source.Update()
            
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(sphere_source.GetOutput())
            
            sphere_actor = vtk.vtkActor()
            sphere_actor.SetMapper(mapper)
            sphere_actor.GetProperty().SetColor(0.0, 1.0, 0.0)  # Green
            scene.add(sphere_actor)
        
        # Create a static time series plot texture
        def create_timeseries_plot():
            fig = Figure(figsize=(8, 3), dpi=100, facecolor='black')
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_subplot(111, facecolor='black')
            
            x = np.arange(len(time_series1))
            ax.plot(x, time_series1, 'b-', linewidth=2, label='Time Series 1')
            ax.plot(x, time_series2, 'r-', linewidth=2, label='Time Series 2')
            
            mean_value1 = np.mean(time_series1)
            mean_value2 = np.mean(time_series2)
            ax.axhline(y=mean_value1, color='cyan', linestyle='--', alpha=0.7)
            ax.axhline(y=mean_value2, color='magenta', linestyle='--', alpha=0.7)
            ax.text(len(time_series1) * 0.02, mean_value1 * 1.05, f'Mean 1: {mean_value1:.2f}', 
                   bbox=dict(facecolor='white', alpha=0.7))
            ax.text(len(time_series2) * 0.02, mean_value2 * 1.05, f'Mean 2: {mean_value2:.2f}', 
                   bbox=dict(facecolor='white', alpha=0.7))
            
            ax.set_title('Time Series Data', fontsize=14, color='white')
            ax.set_xlabel('Time Point', fontsize=12, color='white')
            ax.set_ylabel('Value', fontsize=12, color='white')
            ax.grid(True, alpha=0.3, color='white')
            ax.legend(loc='upper right', fontsize=12, facecolor='black', edgecolor='white', labelcolor='white')
            
            fig.tight_layout()
            canvas.draw()
            
            w, h = canvas.get_width_height()
            buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            buf.shape = (h, w, 3)
            
            img_data = vtk.vtkImageData()
            img_data.SetDimensions(w, h, 1)
            img_data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 3)
            
            for i in range(h):
                for j in range(w):
                    for k in range(3):
                        img_data.SetScalarComponentFromDouble(j, h-i-1, 0, k, buf[i, j, k])
            
            return img_data
        
        plot_plane = vtk.vtkPlaneSource()
        
        y_offset = min_bounds[1] - 150
        plot_plane.SetOrigin(-200, y_offset, -100)
        plot_plane.SetPoint1(200, y_offset, -100)
        plot_plane.SetPoint2(-200, y_offset, 0)
        plot_plane.SetResolution(30, 30)
        plot_plane.Update()
        
        plot_texture = vtk.vtkTexture()
        img_data = create_timeseries_plot()
        plot_texture.SetInputData(img_data)
        plot_texture.InterpolateOn()
        
        plot_mapper = vtk.vtkPolyDataMapper()
        plot_mapper.SetInputData(plot_plane.GetOutput())
        
        plot_actor = vtk.vtkActor()
        plot_actor.SetMapper(plot_mapper)
        plot_actor.SetTexture(plot_texture)
        
        scene.add(plot_actor)
        
        marker_source = vtk.vtkSphereSource()
        marker_source.SetRadius(3.0)
        marker_source.SetPhiResolution(12)
        marker_source.SetThetaResolution(12)
        marker_source.Update()
        
        marker_mapper = vtk.vtkPolyDataMapper()
        marker_mapper.SetInputData(marker_source.GetOutput())
        
        marker_actor = vtk.vtkActor()
        marker_actor.SetMapper(marker_mapper)
        marker_actor.GetProperty().SetColor(1.0, 0.0, 0.0)
        
        marker_x_range = 400
        marker_y_position = y_offset
        marker_z_position = -50
        marker_actor.SetPosition(-200, marker_y_position, marker_z_position)
        
        scene.add(marker_actor)
        
        scene.reset_camera()
        scene.set_camera(position=(0, -500, 0),
                         focal_point=(0, 0, 0),
                         view_up=(0, 0, 1))
        
        # Create a show manager
        show_manager = window.ShowManager(scene,
                                         size=(1200, 800),
                                         title="Brain Visualization with Time Series")
        
        # Disable rotation
        interactor = show_manager.window.GetInteractor()
        interactor_style = vtk.vtkInteractorStyleTrackballCamera()
        interactor.SetInteractorStyle(interactor_style)
        
        print(f"Animating path for ROIs: {roi_indices}")
        print(f"Time series length: {len(time_series1)}")
        
        position = 0
        
        # Initialize streamlines actor
        filtered_streamlines = filter_streamlines(streamlines, roi_coords, roi_sizes)
        streamlines_actor = actor.line(filtered_streamlines)
        streamlines_actor.GetProperty().SetOpacity(0.1)
        scene.add(streamlines_actor)
        
        def timer_callback(_obj, _event):
            nonlocal position
            nonlocal filtered_streamlines
            nonlocal streamlines_actor
            
            position = (position + 1) % len(time_series1)
            
            # Update visibility of streamlines based on time series threshold
            if time_series1[position] > threshold or time_series2[position] > threshold:
                filtered_streamlines = filter_streamlines(streamlines, roi_coords, roi_sizes)
            else:
                filtered_streamlines = []
            
            # Remove the old streamlines actor
            scene.rm(streamlines_actor)
            
            # Create a new streamlines actor with updated data
            streamlines_actor = actor.line(filtered_streamlines)
            streamlines_actor.GetProperty().SetOpacity(0.1)
            scene.add(streamlines_actor)
            
            # Update the marker position on the time series plot
            marker_x = -200 + (position / (len(time_series1) - 1)) * marker_x_range
            marker_actor.SetPosition(marker_x, marker_y_position, marker_z_position)
            
            show_manager.render()
            
            time.sleep(0.1)
        
        show_manager.add_timer_callback(True, 100, timer_callback)
        show_manager.start()
        
    except Exception as e:
        print(f"Error loading or processing data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    trk_file_path = "HCP.trk"
    node_file = "Node_AAL90.node"
    edge_file = "edge_AAL90_binary.edge"
    
    x = np.linspace(0, 4*np.pi, 100)
    time_series1 = 1.0 + 0.5 * np.sin(x) + 0.3 * np.sin(2.5*x) + 0.1 * np.random.randn(100)
    time_series2 = np.cos(np.linspace(0, 4*np.pi, 100)) + 1
    
    roi_indices = [3, 4]  # Example ROIs (rows in the node file)
    threshold = 1.5  # Example threshold
    
    visualize_trk_with_timeseries(
        trk_file_path, 
        node_file, 
        edge_file, 
        roi_indices=roi_indices,
        threshold=threshold,
        time_series1=time_series1,
        time_series2=time_series2
    )