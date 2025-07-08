import pdal
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.patches as patches
from pathlib import Path
import json
import math
from typing import List, Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Configuration
LAZ_DIRECTORY = Path("./DSM_Data/cleaned")
VIEWPOINT_X = 613417
VIEWPOINT_Y = 6462100
VIEWPOINT_Z = 1.5
SEARCH_RADIUS = 1500.0

class InteractiveLAZViewer:
    def __init__(self, directory: Path, initial_viewpoint: Tuple[float, float, float]):
        self.directory = directory
        self.viewpoint = list(initial_viewpoint)
        self.search_radius = SEARCH_RADIUS
        self.point_data = {}
        self.current_view_mode = '3d'
        self.show_viewpoint = True
        self.show_search_radius = True
        self.point_size = 1
        self.elevation_filter = [None, None]  # [min, max]
        
        # Initialize the viewer
        self.setup_viewer()
        
    def load_sample_data(self, max_files: int = 3, max_points_per_file: int = 50000):
        """Load sample data from LAZ files for interactive viewing."""
        print(f"Loading sample data from up to {max_files} files...")
        
        laz_files = self.find_relevant_files()
        files_to_load = laz_files[:max_files]
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for i, laz_file in enumerate(files_to_load):
            try:
                print(f"Loading {laz_file.name}...")
                
                # Create spatial filter around viewpoint
                buffer = self.search_radius * 1.2
                min_x = self.viewpoint[0] - buffer
                max_x = self.viewpoint[0] + buffer
                min_y = self.viewpoint[1] - buffer
                max_y = self.viewpoint[1] + buffer
                
                pipeline_json = {
                    "pipeline": [
                        {
                            "type": "readers.las",
                            "filename": str(laz_file)
                        },
                        {
                            "type": "filters.crop",
                            "bounds": f"([{min_x}, {max_x}], [{min_y}, {max_y}])"
                        },
                        {
                            "type": "filters.head",
                            "count": max_points_per_file
                        }
                    ]
                }
                
                pipeline = pdal.Pipeline(json.dumps(pipeline_json))
                pipeline.execute()
                points = pipeline.arrays[0]
                
                if len(points) > 0:
                    # Apply circular distance filter
                    distances = np.sqrt((points['X'] - self.viewpoint[0])**2 + 
                                       (points['Y'] - self.viewpoint[1])**2)
                    within_radius = distances <= self.search_radius
                    filtered_points = points[within_radius]
                    
                    if len(filtered_points) > 0:
                        self.point_data[laz_file.name] = {
                            'points': filtered_points,
                            'color': colors[i % len(colors)],
                            'visible': True
                        }
                        print(f"  Loaded {len(filtered_points):,} points")
                    else:
                        print(f"  No points within search radius")
                else:
                    print(f"  No points found in spatial bounds")
                    
            except Exception as e:
                print(f"  Error loading {laz_file.name}: {e}")
        
        if not self.point_data:
            raise ValueError("No point data could be loaded!")
        
        total_points = sum(len(data['points']) for data in self.point_data.values())
        print(f"Total points loaded: {total_points:,}")
        
    def find_relevant_files(self) -> List[Path]:
        """Find LAZ files that might contain relevant data."""
        if not self.directory.exists():
            raise FileNotFoundError(f"Directory not found: {self.directory}")
        
        laz_patterns = ["*.laz", "*.LAZ"]
        laz_files = []
        for pattern in laz_patterns:
            found_files = list(self.directory.glob(pattern))
            laz_files.extend(found_files)
        
        laz_files = sorted(list(set(laz_files)))
        
        if not laz_files:
            raise FileNotFoundError(f"No LAZ files found in {self.directory}")
        
        # Filter by JSON metadata if available
        relevant_files = []
        for laz_file in laz_files:
            json_file = laz_file.with_suffix('.json')
            if json_file.exists():
                try:
                    with open(json_file, 'r') as f:
                        json_data = json.load(f)
                    
                    # Check if file bounds intersect with search area
                    if 'geometry' in json_data and 'coordinates' in json_data['geometry']:
                        coordinates = json_data['geometry']['coordinates'][0]
                        box_coords = coordinates[:-1]
                        
                        x_coords = [coord[0] for coord in box_coords]
                        y_coords = [coord[1] for coord in box_coords]
                        
                        min_x, max_x = min(x_coords), max(x_coords)
                        min_y, max_y = min(y_coords), max(y_coords)
                        
                        # Calculate distance to bounding box
                        closest_x = max(min_x, min(self.viewpoint[0], max_x))
                        closest_y = max(min_y, min(self.viewpoint[1], max_y))
                        distance = np.sqrt((self.viewpoint[0] - closest_x)**2 + 
                                         (self.viewpoint[1] - closest_y)**2)
                        
                        if distance <= self.search_radius:
                            relevant_files.append(laz_file)
                except:
                    relevant_files.append(laz_file)  # Include if JSON read fails
            else:
                relevant_files.append(laz_file)  # Include if no JSON
        
        print(f"Found {len(relevant_files)}/{len(laz_files)} potentially relevant files")
        return relevant_files
    
    def setup_viewer(self):
        """Set up the interactive matplotlib viewer."""
        self.fig = plt.figure(figsize=(16, 12))
        
        # Create main 3D plot
        self.ax_3d = self.fig.add_subplot(2, 2, (1, 2), projection='3d')
        
        # Create 2D top-down view
        self.ax_2d = self.fig.add_subplot(2, 2, 3)
        
        # Create control panel
        self.ax_controls = self.fig.add_subplot(2, 2, 4)
        self.ax_controls.axis('off')
        
        # Set up interactive controls
        self.setup_controls()
        
        # Initial plot
        self.update_plot()
        
    def setup_controls(self):
        """Set up interactive controls."""
        # Create control widgets
        controls_y = 0.9
        
        # Viewpoint controls
        self.ax_controls.text(0.05, controls_y, 'VIEWPOINT CONTROLS', fontsize=12, fontweight='bold')
        controls_y -= 0.08
        
        # X coordinate slider
        self.ax_controls.text(0.05, controls_y, f'X: {self.viewpoint[0]:.0f}', fontsize=10)
        controls_y -= 0.05
        
        # Y coordinate slider  
        self.ax_controls.text(0.05, controls_y, f'Y: {self.viewpoint[1]:.0f}', fontsize=10)
        controls_y -= 0.05
        
        # Z coordinate slider
        self.ax_controls.text(0.05, controls_y, f'Z: {self.viewpoint[2]:.1f}', fontsize=10)
        controls_y -= 0.08
        
        # Search radius control
        self.ax_controls.text(0.05, controls_y, f'Search Radius: {self.search_radius:.0f}m', fontsize=10)
        controls_y -= 0.08
        
        # View mode controls
        self.ax_controls.text(0.05, controls_y, 'VIEW CONTROLS', fontsize=12, fontweight='bold')
        controls_y -= 0.05
        
        # Point size control
        self.ax_controls.text(0.05, controls_y, f'Point Size: {self.point_size}', fontsize=10)
        controls_y -= 0.05
        
        # File visibility controls
        self.ax_controls.text(0.05, controls_y, 'FILE VISIBILITY', fontsize=12, fontweight='bold')
        controls_y -= 0.05
        
        # Instructions
        instructions = """
KEYBOARD CONTROLS:
- Arrow keys: Move viewpoint X/Y
- Page Up/Down: Move viewpoint Z
- +/-: Adjust search radius
- 1-9: Change point size
- Space: Toggle search radius circle
- R: Reset view
- H: Toggle this help

MOUSE CONTROLS:
- Left click + drag: Rotate 3D view
- Right click + drag: Pan view
- Scroll: Zoom in/out
        """
        
        self.ax_controls.text(0.05, controls_y - 0.05, instructions, fontsize=8, 
                             verticalalignment='top', fontfamily='monospace')
        
    def update_plot(self):
        """Update the visualization with current data and settings."""
        # Clear previous plots
        self.ax_3d.clear()
        self.ax_2d.clear()
        
        if not self.point_data:
            self.ax_3d.text(0.5, 0.5, 0.5, 'No data loaded\nPress L to load data', 
                           transform=self.ax_3d.transAxes, ha='center', va='center')
            return
        
        # Plot point data
        for filename, data in self.point_data.items():
            if not data['visible']:
                continue
                
            points = data['points']
            color = data['color']
            
            # Apply elevation filter if set
            if self.elevation_filter[0] is not None or self.elevation_filter[1] is not None:
                z_vals = points['Z']
                mask = np.ones(len(points), dtype=bool)
                if self.elevation_filter[0] is not None:
                    mask &= z_vals >= self.elevation_filter[0]
                if self.elevation_filter[1] is not None:
                    mask &= z_vals <= self.elevation_filter[1]
                points = points[mask]
                
                if len(points) == 0:
                    continue
            
            # 3D plot
            self.ax_3d.scatter(points['X'], points['Y'], points['Z'], 
                              c=color, s=self.point_size, alpha=0.6, label=filename)
            
            # 2D plot (top-down view)
            self.ax_2d.scatter(points['X'], points['Y'], 
                              c=color, s=self.point_size, alpha=0.6, label=filename)
        
        # Add viewpoint marker
        if self.show_viewpoint:
            # 3D viewpoint
            self.ax_3d.scatter(self.viewpoint[0], self.viewpoint[1], self.viewpoint[2], 
                              c='black', s=100, marker='*', label='Viewpoint')
            
            # 2D viewpoint
            self.ax_2d.scatter(self.viewpoint[0], self.viewpoint[1], 
                              c='black', s=100, marker='*', label='Viewpoint')
        
        # Add search radius circle
        if self.show_search_radius:
            circle = patches.Circle((self.viewpoint[0], self.viewpoint[1]), 
                                  self.search_radius, fill=False, color='red', 
                                  linestyle='--', linewidth=2, label='Search Radius')
            self.ax_2d.add_patch(circle)
        
        # Set plot properties
        self.ax_3d.set_xlabel('X')
        self.ax_3d.set_ylabel('Y')
        self.ax_3d.set_zlabel('Z (Elevation)')
        self.ax_3d.set_title('3D Point Cloud View')
        self.ax_3d.legend()
        
        self.ax_2d.set_xlabel('X')
        self.ax_2d.set_ylabel('Y')
        self.ax_2d.set_title('Top-Down View')
        self.ax_2d.legend()
        self.ax_2d.set_aspect('equal')
        self.ax_2d.grid(True, alpha=0.3)
        
        # Update control panel
        self.ax_controls.clear()
        self.ax_controls.axis('off')
        
        status_text = f"""
STATUS:
Viewpoint: ({self.viewpoint[0]:.0f}, {self.viewpoint[1]:.0f}, {self.viewpoint[2]:.1f})
Search Radius: {self.search_radius:.0f}m
Point Size: {self.point_size}
Files Loaded: {len(self.point_data)}

LOADED FILES:
{chr(10).join(f"- {name} ({len(data['points']):,} pts)" for name, data in self.point_data.items())}

KEYBOARD CONTROLS:
Arrow Keys: Move viewpoint X/Y (±10m)
Page Up/Down: Move viewpoint Z (±1m)
+/-: Adjust search radius (±100m)
1-9: Change point size
Space: Toggle search radius circle
V: Toggle viewpoint marker
R: Reset viewpoint
L: Load/reload data
Q: Quit
        """
        
        self.ax_controls.text(0.05, 0.95, status_text, transform=self.ax_controls.transAxes,
                             fontsize=9, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        self.fig.canvas.draw()
    
    def on_key_press(self, event):
        """Handle keyboard input."""
        if event.key == 'up':
            self.viewpoint[1] += 10
        elif event.key == 'down':
            self.viewpoint[1] -= 10
        elif event.key == 'left':
            self.viewpoint[0] -= 10
        elif event.key == 'right':
            self.viewpoint[0] += 10
        elif event.key == 'pageup':
            self.viewpoint[2] += 1
        elif event.key == 'pagedown':
            self.viewpoint[2] -= 1
        elif event.key == '+' or event.key == '=':
            self.search_radius += 100
        elif event.key == '-':
            self.search_radius = max(100, self.search_radius - 100)
        elif event.key in '123456789':
            self.point_size = int(event.key)
        elif event.key == ' ':
            self.show_search_radius = not self.show_search_radius
        elif event.key == 'v':
            self.show_viewpoint = not self.show_viewpoint
        elif event.key == 'r':
            self.viewpoint = [VIEWPOINT_X, VIEWPOINT_Y, VIEWPOINT_Z]
            self.search_radius = SEARCH_RADIUS
        elif event.key == 'l':
            self.load_sample_data()
        elif event.key == 'q':
            plt.close(self.fig)
            return
        
        self.update_plot()
    
    def run(self):
        """Start the interactive viewer."""
        print("Loading initial data...")
        self.load_sample_data()
        
        print("\nStarting interactive viewer...")
        print("Use keyboard controls to navigate:")
        print("- Arrow keys: Move viewpoint X/Y")
        print("- Page Up/Down: Move viewpoint Z")
        print("- +/-: Adjust search radius")
        print("- 1-9: Change point size")
        print("- Space: Toggle search radius circle")
        print("- V: Toggle viewpoint marker")
        print("- R: Reset viewpoint")
        print("- L: Load/reload data")
        print("- Q: Quit")
        
        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Update plot
        self.update_plot()
        
        # Show the plot
        plt.show()

def main():
    """Main function to start the interactive viewer."""
    print("=== INTERACTIVE LAZ POINT CLOUD VIEWER ===")
    
    try:
        viewer = InteractiveLAZViewer(LAZ_DIRECTORY, (VIEWPOINT_X, VIEWPOINT_Y, VIEWPOINT_Z))
        viewer.run()
        
    except Exception as e:
        print(f"Error starting viewer: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()