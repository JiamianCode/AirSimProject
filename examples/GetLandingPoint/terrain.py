import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def generate_dem(world_points, resolution, method='nearest'):
    max_height = 0.75 * max(world_points[:, 2])
    print("Max Height:", max_height)
    filtered_points = world_points[world_points[:, 2] <= max_height]
    
    x = filtered_points[:, 0]
    y = filtered_points[:, 1]
    z = filtered_points[:, 2]
    
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    
    grid_x, grid_y = np.meshgrid(
        np.arange(x_min, x_max, resolution),
        np.arange(y_min, y_max, resolution)
    )
    
    grid_z = griddata((x, y), z, (grid_x, grid_y), method=method)
    
    return grid_x, grid_y, grid_z

world_points = np.loadtxt("elevation_points.csv", delimiter=",")
world_points[:, 1] = -world_points[:, 1]
world_points[:, 2] = -world_points[:, 2]



fig = plt.figure(figsize=(15, 6))

ax1 = fig.add_subplot(121, projection='3d')
sc = ax1.scatter(world_points[:, 0], world_points[:, 1], world_points[:, 2], 
                 c=world_points[:, 2], cmap='terrain', marker='o', s=1)
plt.colorbar(sc, ax=ax1, label="Elevation (mapped)")
ax1.set_xlabel("Mapped X")
ax1.set_ylabel("Mapped Y")
ax1.set_zlabel("Mapped Z")
ax1.set_title("3D Scatter Plot")

grid_x, grid_y, grid_z = generate_dem(world_points, resolution=0.1, method='nearest')

# shape of grid_x, grid_y, grid_z
print(grid_x.shape, grid_y.shape, grid_z.shape)
with open("shape_data.csv", "w") as f:
    f.write(f"{grid_x.shape[0]},{grid_x.shape[1]}")

# store to csv file
xyz_data = np.column_stack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel()))
np.savetxt("DEM_data.csv", xyz_data, delimiter=",")

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(grid_x, grid_y, grid_z, cmap="terrain", edgecolor='none')
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Height")
ax2.set_title("Generated 3D Terrain")

plt.show()
