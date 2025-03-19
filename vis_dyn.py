import numpy as np
from fury import window, actor
from dipy.io.streamline import load_trk
import os
import vtk
from dipy.tracking.streamline import set_number_of_points

def vis_effconn(node_file, edge_file, trk_file_path="HCP.trk", show_tractography=True):
    """Animate a traveling gradient along connected tracts."""
    # File validation
    for file_path in [trk_file_path, node_file, edge_file]:
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} does not exist.")
            return

    scene = window.Scene()
    streamlines = None
    gradient_actor = None
    phase = 0.0  # Animation control variable

    try:
        # Load tractography data
        if show_tractography:
            tractogram = load_trk(trk_file_path, reference="same", bbox_valid_check=False)
            streamlines = tractogram.streamlines
            all_points = np.vstack(streamlines)
            tract_center = (np.min(all_points, axis=0) + np.max(all_points, axis=0)) / 2
        else:
            tract_center = np.zeros(3)

        # Load and adjust node positions (unchanged)
        with open(node_file, 'r') as f:
            node_data = [line.strip().split() for line in f if len(line.strip().split()) >= 6]
        
        node_coords = np.array([[float(p[0]), float(p[1]), float(p[2])] for p in node_data])
        node_sizes = np.array([float(p[4]) for p in node_data])
        
        if len(node_coords) != 90:
            print("AAL90 requires 90 nodes.")
            return

        # Adjust node positions to tractography center
        node_center = (np.min(node_coords, axis=0) + np.max(node_coords, axis=0)) / 2
        node_coords += tract_center - node_center

        # Create node actors (unchanged)
        node_assembly = vtk.vtkAssembly()
        for coord, size in zip(node_coords, node_sizes):
            sphere = vtk.vtkSphereSource()
            sphere.SetCenter(coord)
            sphere.SetRadius(size * 0.9)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.8, 0.3, 0.3)
            node_assembly.AddPart(actor)
        scene.add(node_assembly)

        # Load connectivity matrix
        connectivity = np.loadtxt(edge_file)
        np.fill_diagonal(connectivity, 0)

        # Process streamlines
        if show_tractography and streamlines:
            print("Creating animated gradient tracts...")
            resampled_streamlines = []
            all_t_values = []
            proximity_radii = node_sizes * 15

            # Find connected streamlines
            connected_streamlines = []
            for sl in streamlines:
                if len(sl) < 2:
                    continue
                
                start_node = end_node = -1
                for i, (pos, rad) in enumerate(zip(node_coords, proximity_radii)):
                    if np.linalg.norm(sl[0] - pos) < rad:
                        start_node = i
                    if np.linalg.norm(sl[-1] - pos) < rad:
                        end_node = i
                
                if start_node != end_node and connectivity[start_node, end_node] > 0:
                    connected_streamlines.append(sl)

            # Prepare animation data
            for sl in connected_streamlines:
                rs = set_number_of_points(sl, 40)
                resampled_streamlines.append(rs)
                all_t_values.extend(np.linspace(0, 1, len(rs)))

            # Create VTK pipeline
            polydata = vtk.vtkPolyData()
            points = vtk.vtkPoints()
            lines = vtk.vtkCellArray()
            colors = vtk.vtkUnsignedCharArray()
            colors.SetNumberOfComponents(3)
            
            point_idx = 0
            for sl in resampled_streamlines:
                line = vtk.vtkPolyLine()
                line.GetPointIds().SetNumberOfIds(len(sl))
                for i, pt in enumerate(sl):
                    points.InsertNextPoint(pt)
                    line.GetPointIds().SetId(i, point_idx)
                    point_idx += 1
                lines.InsertNextCell(line)
            
            polydata.SetPoints(points)
            polydata.SetLines(lines)
            
            # Initial colors
            colors_array = np.ones((point_idx, 3)) * 255
            for color in colors_array:
                colors.InsertNextTuple3(*color.astype(int))
            polydata.GetPointData().SetScalars(colors)

            # Create mapper/actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(polydata)
            gradient_actor = vtk.vtkActor()
            gradient_actor.SetMapper(mapper)
            gradient_actor.GetProperty().SetOpacity(0.9)
            gradient_actor.GetProperty().SetLineWidth(2)
            
            # Store animation parameters
            gradient_actor.t_values = np.array(all_t_values)
            gradient_actor.polydata = polydata
            gradient_actor.colors = colors
            scene.add(gradient_actor)

        # Camera setup
        scene.reset_camera()
        scene.set_camera(position=(0, -500, 0), focal_point=(0, 0, 0), view_up=(0, 0, 1))

        # Create show manager
        show_manager = window.ShowManager(scene, size=(1024, 768), 
                                        title="Traveling Gradient Animation")

        # Animation callback
        def update_colors(_obj, _event):
            nonlocal phase
            phase = (phase + 0.02) % 1.0  # Controls animation speed
            
            if gradient_actor:
                t = gradient_actor.t_values
                # Create moving gradient effect
                gradient_pos = (t + phase) % 1.0
                
                # Smooth transition between colors using sine wave
                white_intensity = np.sin(gradient_pos * np.pi) ** 2
                
                # Blue (0,0,1) to white (1,1,1) transition
                r = white_intensity
                g = white_intensity
                b = np.ones_like(t)  # Fixed: Create array with same shape as t
                
                # Combine and scale to 0-255
                colors_array = np.stack([r, g, b], axis=1) * 255
                
                # Update VTK colors
                gradient_actor.colors.Reset()
                for color in colors_array.astype(np.uint8):
                    gradient_actor.colors.InsertNextTuple3(*color)
                
                gradient_actor.polydata.Modified()
                show_manager.render()

        show_manager.add_timer_callback(True, 30, update_colors)
        show_manager.start()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    vis_effconn("Node_AAL90.node", "edge_AAL90_binary.edge", "HCP.trk")