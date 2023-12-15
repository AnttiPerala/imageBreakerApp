from scipy.spatial import Voronoi, voronoi_plot_2d
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# Function to generate Voronoi points with full coverage
def generate_voronoi_points_full_coverage(num_points, image_size):
    width, height = image_size
    corners = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
    num_border_points = num_points // 2  # Let's allocate half of the points to border points for safety
    border_points = np.vstack((
        np.column_stack((np.random.randint(0, width, num_border_points // 4), np.zeros(num_border_points // 4))),
        np.column_stack((np.random.randint(0, width, num_border_points // 4), np.full(num_border_points // 4, height-1))),
        np.column_stack((np.zeros(num_border_points // 4), np.random.randint(0, height, num_border_points // 4))),
        np.column_stack((np.full(num_border_points // 4, width-1), np.random.randint(0, height, num_border_points // 4))),
    ))
    # Ensure we do not attempt to create a negative number of inner points
    num_inner_points = max(num_points - len(corners) - len(border_points), 0)
    inner_points = np.column_stack((
        np.random.randint(1, width-1, num_inner_points),
        np.random.randint(1, height-1, num_inner_points)
    ))
    return np.vstack((corners, border_points, inner_points))



# Function to assign pixels to cells using a k-d tree
def assign_pixels_to_cells_kdtree(img, points):
    tree = cKDTree(points)
    width, height = img.size
    cells = {i: [] for i in range(len(points))}

    for x in range(width):
        for y in range(height):
            dist, closest_point_idx = tree.query([x, y], k=1)
            cells[closest_point_idx].append((x, y))

    return cells

# Function to create cell images
def create_cell_images(img, cells):
    width, height = img.size
    cell_images = []

    for cell_index, cell_pixels in cells.items():
        cell_img = Image.new('RGBA', (width, height), (255, 255, 255, 0))
        pixels = cell_img.load()

        for x, y in cell_pixels:
            if 0 <= x < width and 0 <= y < height:
                pixels[x, y] = img.getpixel((x, y))

        cell_images.append(cell_img)

    return cell_images

# Main function to process the image with full coverage Voronoi points
def process_image_with_full_coverage_voronoi(image_path, num_points):
    img = Image.open(image_path).convert('RGBA')
    width, height = img.size
    points = generate_voronoi_points_full_coverage(num_points, (width, height))
    vor = Voronoi(points)

    cells = assign_pixels_to_cells_kdtree(img, points)
    cell_images = create_cell_images(img, cells)

   # Optionally save cell images and plot for demonstration
    output_paths = []
    for i, cell_img in enumerate(cell_images):
        cell_img_path = f'./cell_full_coverage_{i}.png'  # Save in the current working directory
        cell_img.save(cell_img_path)
        output_paths.append(cell_img_path)

    # Plotting the Voronoi diagram for visualization
    plt.figure(figsize=(8, 12))
    voronoi_plot_2d(vor, show_vertices=False, point_size=1)
    plt.xlim(0, width)
    plt.ylim(0, height)
    plt.gca().invert_yaxis()
    plt.show()

    return output_paths, points

# Set the path to the image and define the number of points
image_path = 'coconut.png'  # Replace with the correct path if not running in this notebook environment
num_points = 100

# Call the main function to process the image
cell_image_paths_full_coverage, voronoi_points_full_coverage = process_image_with_full_coverage_voronoi(image_path, num_points)

