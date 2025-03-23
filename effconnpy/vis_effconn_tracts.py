import numpy as np
from fury import window, actor
from dipy.io.streamline import load_trk
import os
import vtk
from dipy.tracking.streamline import set_number_of_points
from dipy.tracking.streamline import length as sl_length
import colorsys

def vis_effconn(node_file, edge_file, trk_file_path="HCP.trk", show_tractography=True, show_background_tracts=False):
    """
    Load and visualize a .trk file with overlay to a connectome atlas defining the directionality,
    with gradient shading of tracts based on region connectivity.

    Parameters
    ----------
    trk_file_path : str
        Path to the .trk file
    node_file : str
        Path to the node coordinates file
    edge_file : str
        Path to the edge/connectivity matrix file
    show_tractography : bool, optional
        Whether to show the tractography data. Default is True.
    """
    # Check if files exist
    for file_path in [trk_file_path, node_file, edge_file]:
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} does not exist.")
            return

    # Create a scene
    scene = window.Scene()
    
    #scene.SetBackground(1.0, 1.0, 1.0) 

    # Initialize streamlines_actor as None
    streamlines_actor = None
    streamlines = None
    streamlines_data = None

    try:
        # Get the center of tracts to align nodes properly
        if show_tractography:
            # Load the tractogram with bbox_valid_check=False to bypass validation
            print(f"Loading {trk_file_path}...")
            tractogram = load_trk(trk_file_path, reference="same", bbox_valid_check=False)

            # Get streamlines in their current space
            streamlines = tractogram.streamlines

            # Get the bounding box of the streamlines to help with placement
            all_points = np.vstack([s for s in streamlines])
            min_bounds = np.min(all_points, axis=0)
            max_bounds = np.max(all_points, axis=0)
            center_of_tracts = (min_bounds + max_bounds) / 2
            print(f"Tract bounds: Min {min_bounds}, Max {max_bounds}, Center {center_of_tracts}")
        else:
            # If not showing tractography, we'll use a default center or the center of nodes
            center_of_tracts = np.zeros(3)
            print("Tractography display disabled.")

        # Load node coordinates
        print(f"Loading {node_file}...")

        # Read the node file as plain text and process it manually
        with open(node_file, 'r') as f:
            lines = f.readlines()

        # Initialize lists to store data
        node_coords = []
        node_sizes = []
        node_names = []  # Initialize node_names as an empty list

        # Process each line
        for line in lines:
            try:
                # Split the line by tabs or spaces
                parts = line.strip().split()

                # Ensure the line has at least 6 values (x, y, z, color, size, name)
                if len(parts) < 6:
                    print(f"Skipping invalid line (expected at least 6 values): {line.strip()}")
                    continue

                # Process the values: coordinates, color, size, and name
                x = float(parts[0])
                y = float(parts[1])
                z = float(parts[2])
                color = float(parts[3])  # Color is the fourth value (unused in this code)
                size = float(parts[4])  # Size is the fifth value
                name = parts[5]  # Name is the sixth value

                # Append coordinates, size, and name to the respective lists
                node_coords.append([x, y, z])
                node_sizes.append(size)
                node_names.append(name)

            except ValueError as e:
                print(f"Skipping invalid line (could not parse coordinates, color, size, or name): {line.strip()}")

        # Convert to numpy arrays
        node_coords = np.array(node_coords, dtype=np.float64)
        node_sizes = np.array(node_sizes, dtype=np.float64)

        # Debugging: Print the number of nodes loaded
        print(f"Loaded {len(node_coords)} nodes.")
        if len(node_coords) == 0:
            print("Error: No valid nodes were loaded. Check the node file format.")
            return  # Exit the function if no nodes are loaded

        # Ensure exactly 90 nodes are loaded for AAL90 atlas
        if len(node_coords) != 90:
            print(f"Error: Expected 90 nodes (AAL90 atlas), but found {len(node_coords)} nodes.")
            return

        # Calculate the bounding box and center of the connectome
        min_node_bounds = np.min(node_coords, axis=0)
        max_node_bounds = np.max(node_coords, axis=0)
        center_of_nodes = (min_node_bounds + max_node_bounds) / 2
        print(f"Node bounds: Min {min_node_bounds}, Max {max_node_bounds}, Center {center_of_nodes}")

        # If we don't have tractography and thus no center_of_tracts, use the nodes center
        if not show_tractography:
            center_of_tracts = center_of_nodes

        # Calculate the translation needed to align centers
        translation = center_of_tracts - center_of_nodes
        print(f"Translation vector: {translation}")

        # Apply translation to node coordinates
        adjusted_node_coords = node_coords + translation

        # Load connectivity matrix
        print(f"Loading {edge_file}...")
        connectivity_matrix = np.loadtxt(edge_file)
        
        # Number of elements to modify
        n = 90  # Matrix size
        # Number of elements to remove asymmetrically
        num_changes = int(0.1 * n * (n - 1))  # Modify 20% of off-diagonal elements

        # Get upper triangular indices (excluding diagonal)
        upper_tri_indices = np.triu_indices(n, k=1)  # k=1 excludes diagonal
        chosen_indices = np.random.choice(len(upper_tri_indices[0]), num_changes, replace=False)

        # Modify selected elements
        for idx in chosen_indices:
            i, j = upper_tri_indices[0][idx], upper_tri_indices[1][idx]
            connectivity_matrix[i, j] = 0  # Set to 0 asymmetrically (but not A[j, i])


        # Ensure the connectivity matrix is 90x90
        if connectivity_matrix.shape != (90, 90):
            print(f"Error: Connectivity matrix must be 90x90 for AAL90 atlas, but found shape {connectivity_matrix.shape}.")
            return

        # Create containers for different elements of the visualization
        node_assembly = vtk.vtkAssembly()

        # Lists to store actors
        node_actors = []

        # Create manual spheres for nodes
        for i, (coord, size) in enumerate(zip(adjusted_node_coords, node_sizes)):
            # Create sphere source
            sphere_source = vtk.vtkSphereSource()
            sphere_source.SetCenter(coord[0], coord[1], coord[2])
            sphere_source.SetRadius(size * 0.9)  # Scale size for better visibility
            sphere_source.SetPhiResolution(12)
            sphere_source.SetThetaResolution(12)
            sphere_source.Update()

            # Create mapper
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(sphere_source.GetOutput())

            # Create actor
            sphere_actor = vtk.vtkActor()
            sphere_actor.SetMapper(mapper)
            sphere_actor.GetProperty().SetColor(0.8, 0.3, 0.3)  # Red color

            # Add to scene
            scene.add(sphere_actor)
            node_assembly.AddPart(sphere_actor)
            node_actors.append(sphere_actor)

        # Add the node assembly to the scene
        scene.add(node_assembly)

        # Define a helper function to check if a point is near a node
        def is_point_near_node(point, node_pos, radius):
            """Check if a point is within radius of a node"""
            distance = np.linalg.norm(point - node_pos)
            return distance < radius
        
        # Function to resample streamlines to have exactly N points
        def resample_streamline(streamline, n_points=60):
            """Resample a streamline to have exactly n_points."""
            try:
                return set_number_of_points(streamline, n_points)
            except:
                # If set_number_of_points fails, do simple linear interpolation
                points = np.array(streamline)
                if len(points) < 2:
                    return points
                
                # Create new points array with n_points
                new_points = np.zeros((n_points, 3))
                
                # Calculate normalized distances along the streamline
                cumulative_distances = np.zeros(len(points))
                for i in range(1, len(points)):
                    cumulative_distances[i] = cumulative_distances[i-1] + np.linalg.norm(points[i] - points[i-1])
                
                if cumulative_distances[-1] == 0:
                    # If streamline has zero length, return repeated points
                    return np.tile(points[0], (n_points, 1))
                
                normalized_distances = cumulative_distances / cumulative_distances[-1]
                
                # Create evenly spaced points
                new_normalized_distances = np.linspace(0, 1, n_points)
                
                # Interpolate for each dimension
                for dim in range(3):
                    new_points[:, dim] = np.interp(new_normalized_distances, normalized_distances, points[:, dim])
                
                return new_points

        # Now we'll identify and color streamlines based on the connectivity matrix
        if show_tractography and streamlines is not None:
            print("Analyzing and creating gradient-shaded streamlines based on connectivity matrix...")
            
            # Calculate node radii for streamline proximity detection
            proximity_radii = node_sizes * 15  # Adjust this multiplier as needed
            
            # Lists to store processed streamline data
            resampled_streamlines = []
            streamline_colors = []
            
            # Track which streamlines are connected
            connected_streamlines = []
            background_streamlines = []
            
            # Fixed number of points for each resampled streamline
            n_points = 40
            
            # Iterate through all streamlines
            for i, streamline in enumerate(streamlines):
                # Skip streamlines that are too short
                if len(streamline) < 2:
                    continue
                
                # Find which nodes the streamline connects
                start_near_node = -1
                end_near_node = -1
                
                # Check first and last points of streamline against all nodes
                for node_idx, (node_pos, radius) in enumerate(zip(adjusted_node_coords, proximity_radii)):
                    # Check if start point is near this node
                    if is_point_near_node(streamline[0], node_pos, radius) and start_near_node == -1:
                        start_near_node = node_idx
                    
                    # Check if end point is near this node
                    if is_point_near_node(streamline[-1], node_pos, radius) and end_near_node == -1:
                        end_near_node = node_idx
                
                # If both endpoints are near different nodes and there's connectivity between them
                if (start_near_node != -1 and end_near_node != -1 and 
                    start_near_node != end_near_node and 
                    connectivity_matrix[start_near_node, end_near_node] > 0):
                    
                    # Resample the streamline to have exactly n_points
                    try:
                        resampled = resample_streamline(streamline, n_points)
                        
                        # Create gradient colors along the streamline
                        # Start with red (for start node) and transition to blue (for end node)
                        colors = np.zeros((n_points, 3))
                        
                        # Gradient from red to blue
                        for j in range(n_points):
                            t = j / (n_points - 1)  # Normalized position [0, 1]
                            
                            # Linear interpolation between red and blue
                            r = t
                            g = t
                            b = 1   # Starting from blue (b = 1) and going to white (b = 1)
                            
                            colors[j] = [r, g, b]
                        
                        # Store this streamline and its colors
                        resampled_streamlines.append(resampled)
                        streamline_colors.append(colors)
                        connected_streamlines.append(streamline)
                    except Exception as e:
                        print(f"Error processing streamline: {e}")
                else:
                    background_streamlines.append(streamline)
            
            print(f"Found {len(connected_streamlines)} connected streamlines and {len(background_streamlines)} background streamlines")
            
            # Display background streamlines with low opacity
            if background_streamlines:
             if show_background_tracts:
                background_actor = actor.line(background_streamlines)
                background_actor.GetProperty().SetOpacity(0.03)  # Very transparent
                background_actor.GetProperty().SetColor(0.5, 0.5, 0.5)  # Grey
                scene.add(background_actor)
            
            # Display gradient-colored streamlines using custom VTK pipeline
            if resampled_streamlines:
                print("Creating gradient-colored streamlines...")
                
                # Create a polydata object to store all gradient-colored streamlines
                polydata = vtk.vtkPolyData()
                points = vtk.vtkPoints()
                lines = vtk.vtkCellArray()
                colors = vtk.vtkUnsignedCharArray()
                colors.SetNumberOfComponents(3)
                colors.SetName("Colors")
                
                point_idx = 0
                
                # Add each streamline as a separate line cell with its own color array
                for sl_idx, (streamline, color_array) in enumerate(zip(resampled_streamlines, streamline_colors)):
                    # Start a new line
                    line = vtk.vtkPolyLine()
                    line.GetPointIds().SetNumberOfIds(len(streamline))
                    
                    # Add points and colors
                    for i, (point, color) in enumerate(zip(streamline, color_array)):
                        points.InsertNextPoint(point[0], point[1], point[2])
                        line.GetPointIds().SetId(i, point_idx)
                        # Convert floating point colors [0,1] to unsigned char [0,255]
                        r, g, b = [int(c * 255) for c in color]
                        colors.InsertNextTuple3(r, g, b)
                        point_idx += 1
                    
                    # Add this line to the cell array
                    lines.InsertNextCell(line)
                
                # Set the points and cells in polydata
                polydata.SetPoints(points)
                polydata.SetLines(lines)
                
                # Add colors to the points
                polydata.GetPointData().SetScalars(colors)
                
                # Create mapper and actor
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(polydata)
                
                # Create actor
                gradient_actor = vtk.vtkActor()
                gradient_actor.SetMapper(mapper)
                gradient_actor.GetProperty().SetOpacity(0.8)  # More visible
                gradient_actor.GetProperty().SetLineWidth(1.5)  # Slightly thicker lines
                
                # Add to scene
                scene.add(gradient_actor)
        
        # Add orientation markers
        scene.add(actor.axes())

        # Reset camera
        scene.reset_camera()
        scene.set_camera(position=(0, -500, 0),
                         focal_point=(0, 0, 0),
                         view_up=(0, 0, 1))

        # Create a show manager
        show_manager = window.ShowManager(scene,
                                        size=(1024, 768),  # Larger window
                                        title="Gradient-Shaded Tractography and Connectivity Visualization")

        # Report information
        if show_tractography:
            if streamlines is not None:
                print(f"Rendering {len(streamlines)} streamlines and {len(node_coords)} atlas nodes")
                print(f"Displaying {len(resampled_streamlines)} as gradient-colored connected fibers")
            else:
                print(f"No streamlines found to render")
        else:
            print(f"Rendering {len(node_coords)} atlas nodes (tractography disabled)")

        # Start the visualization
        show_manager.start()

    except Exception as e:
        print(f"Error loading or processing data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Replace with paths to your files
    trk_file_path = "HCP.trk"  # Updated to match your file
    node_file = "Node_AAL90.node"
    edge_file = "edge_AAL90_binary.edge"

    # To show both tractography and graphs (default):
    vis_effconn(node_file, edge_file, trk_file_path, show_tractography=True)

    # To show only the graphs (nodes and connections):
    # vis_effconn(node_file, edge_file, trk_file_path, show_tractography=False)
