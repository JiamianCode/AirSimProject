import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# import alphashape
from scipy.spatial import distance
# from shapely.geometry import MultiPolygon


import sys
sys.setrecursionlimit(2000)

# global variables
connected_points_set = []
connected_points_cnt = 0



# Function for the Planarity part
def find_connected_points(steppability, z, start_x, start_y, z_threshold=0.1):
    rows, cols = steppability.shape
    global visited
    global connected_points_set
    global connected_points_cnt
    
    if visited[start_y, start_x] != 0:
        return connected_points_set[int(visited[start_y, start_x] - 1)]
    
    connected_points = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    connected_points_cnt += 1

    def dfs(x, y):
        if x < 0 or x >= cols or y < 0 or y >= rows or (visited[y, x] != 0) or steppability[y, x] == 0:
            return
        visited[y, x] = connected_points_cnt
        connected_points.append((x, y, z[y, x]))
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < cols and 0 <= ny < rows and (visited[ny, nx] == 0):
                if abs(z[y, x] - z[ny, nx]) <= z_threshold:
                    dfs(nx, ny)

    dfs(start_x, start_y)
    connected_points_set.append(connected_points)
    return connected_points

def find_best_landing_point(region_points, judging):
    best_score = -1
    best_point = None
    boundary_points = []
    candidate_points = []
    
    for rpoint in region_points:
        x, y = rpoint[0], rpoint[1]
        boundary_density = 0
        for i in range(x-1, x+2):
            for j in range(y-1, y+2):
                if i >= 0 and i < x_num and j >= 0 and j < y_num and judging[j][i] == 1:
                    boundary_density += 1
        if boundary_density <= 6:
            boundary_points.append(rpoint)
        else:
            candidate_points.append(rpoint)
    # find the best point which is farthest from the closest boundary
    for cpoint in candidate_points:
        min_distance = 1000
        for bpoint in boundary_points:
            dist = distance.euclidean(cpoint, bpoint)
            if dist < min_distance:
                min_distance = dist
        score = min_distance
        if score > best_score:
            best_score = score
            best_point = cpoint
    if best_score == -1 or best_score == 1000:
        return None, None
    return best_point, best_score


with open("shape_data.csv", "r") as f:
    rlines = f.readlines()
shape_y_num, shape_x_num = rlines[0].split(",")
print(f"Shape of DEM data is x = {shape_x_num} and y = {shape_y_num}")
x_num = int(shape_x_num)
y_num = int(shape_y_num)


# Load the terrain data and store x and y coordinates
coordinates = []
x_coordinates = np.zeros((y_num, x_num))
y_coordinates = np.zeros((y_num, x_num))
z = np.zeros((y_num, x_num))
i = 0
j = 0
with open("DEM_data.csv", mode="r", encoding="utf-8") as file:
    reader = csv.reader(file)
    for row in reader:
        coordinates.append(row)
        x_coordinates[i, j] = float(row[0])
        y_coordinates[i, j] = float(row[1])
        z[i, j] = float(row[2])
        j += 1
        if j == x_num:
            j = 0
            i += 1
        if i == y_num:
            break

# Plot the 3D terrain
x = np.arange(0, x_num, 1)
y = np.arange(0, y_num, 1)
x, y = np.meshgrid(x, y)
fig = plt.figure(figsize=(15, 12))
ax = fig.add_subplot(131, projection='3d')
ax.plot_surface(x, y, z, cmap="terrain", edgecolor='none')
ax.set_title("Generated 3D Terrain with Random Noises")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Height")
# plt.show()

# Steppability and Planarity
N = 9
step_cnt = 0
# eigenvalue_parameter = 0.2
# eigenvector_parameter = 0.6
# eigenvalue_parameter_p = 0.025
# eigenvector_parameter_p = 0.87
eigenvalue_parameter = 0.018
eigenvector_parameter = 0.9962
eigenvalue_parameter_p = 0.02
eigenvector_parameter_p = 0.99
steppability = np.zeros((y_num, x_num))
for i in range(x_num):
    for j in range(y_num):
        # surrounding 9 points
        points = []
        if i == 0 or i == (x_num - 1) or j == 0 or j == (y_num - 1):
            continue
        for x in range(i-1, i+2):
            for y in range(j-1, j+2):
                if x >= 0 and x < (x_num) and y >= 0 and y < (y_num):
                    points.append([x, y, z[y, x]])
        # center point
        center = np.mean(points, axis=0)
        # covariance
        points_centered = points - center
        cov_matrix = np.cov(points_centered, rowvar=False)
        # eigenvalue and eigenvector
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        # corresponding eigenvalue and eigenvector
        min_eigenvalue_index = np.argmin(eigenvalues)
        min_eigenvalue = eigenvalues[min_eigenvalue_index]
        min_eigenvector = eigenvectors[:, min_eigenvalue_index]
        min_comp_temp1 = abs(min_eigenvector[2])
        # first step for steppability
        if min_eigenvalue <= eigenvalue_parameter and min_comp_temp1 >= eigenvector_parameter:
            steppability[j][i] = 1
            step_cnt += 1
        # else:
        #     print("value is", min_eigenvalue, "vector is", min_eigenvector[2], "at", i, j)
print(f"Steppable points out of {y_num*x_num}:", step_cnt)

step_cnt = 0
triangles = []
min_tsize = 50
visited = np.zeros_like(steppability)
planarity_visited = np.zeros_like(steppability)
# Planarity Process
for i in range(x_num):
    for j in range(y_num):
        if steppability[j][i] == 0:
            continue
        if planarity_visited[j][i] == 1:
            continue
        connected_points = find_connected_points(steppability, z, i, j)
        lenN = len(connected_points)
        if lenN < min_tsize:
            for point_tmp in connected_points:
                steppability[point_tmp[1]][point_tmp[0]] = 0
                planarity_visited[int(point_tmp[1])][int(point_tmp[0])] = 1
            continue
        # planarity judgement
        connected_points = np.array(connected_points)
        center = np.mean(connected_points, axis=0)
        points_centered = connected_points - center
        cov_matrix = np.cov(points_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        min_eigenvalue_index = np.argmin(eigenvalues)
        min_eigenvalue = eigenvalues[min_eigenvalue_index]
        min_eigenvector = eigenvectors[:, min_eigenvalue_index]
        min_comp_temp2 = abs(min_eigenvector[2])
        if min_eigenvalue > eigenvalue_parameter_p or min_comp_temp2 < eigenvector_parameter_p:
            for point_tmp in connected_points:
                steppability[point_tmp[1]][point_tmp[0]] = 0
                planarity_visited[int(point_tmp[1])][int(point_tmp[0])] = 1
        else:
            for point_tmp in connected_points:
                step_cnt += 1
                planarity_visited[int(point_tmp[1])][int(point_tmp[0])] = 1

print(f"Steppable points out of {y_num*x_num} after planarity:", step_cnt)
print("Steppable regions found:", connected_points_cnt)    


# 3D terrain considering steppability
x, y = np.meshgrid(np.arange(0, x_num, 1), np.arange(0, y_num, 1))
x_filtered = x[steppability == 1]
y_filtered = y[steppability == 1]
z_filtered = z[steppability == 1]
# fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(132, projection='3d')
# ax.plot_surface(x, y, z, cmap="terrain", edgecolor='none', alpha=0.5)
ax.scatter(x_filtered, y_filtered, z_filtered, c='r', marker='o', s = 5)
ax.set_title("3D Terrain with Steppability = 1 Points")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
# plt.show()


density = np.zeros((y_num, x_num))
density_cnt = 0
# Density filtering
for i in range(x_num):
    for j in range(y_num):
        if i == 0 or i == (x_num - 1) or j == 0 or j == (y_num - 1) or steppability[j][i] == 0:
            continue
        steppable = 0
        for x in range(i-1, i+2):
            for y in range(j-1, j+2):
                if x >= 0 and x < x_num and y >= 0 and y < y_num and steppability[y][x] == 1 and (abs(z[y][x] - z[j][i])<=0.25):
                    steppable += 1
        if steppable < 8:
            continue
        density[j][i] = 1
        # print(i, j, z[j][i])
        density_cnt += 1
        
        
# find connected regions with density
connected_points_set = []
connected_points_cnt = 0
visited = np.zeros_like(density)
for i in range(x_num):
    for j in range(y_num):
        if density[j][i] == 0 or visited[j, i] != 0:
            continue
        find_connected_points(density, z, i, j)
        
        
print(f"Steppable points out of {y_num*x_num} after density:", density_cnt)
print("Steppable regions found:", connected_points_cnt)


# find an optimal landing point of each region with a score
global_best_score = -1
global_best_point = None
for i in range(connected_points_cnt):
    print("Region", i, "has", len(connected_points_set[i]), "points")
    if len(connected_points_set[i]) < min_tsize:
        continue
    else:
        print("Drawing region", i)
    
    current_point, current_score = find_best_landing_point(connected_points_set[i], density)
    
    if current_point == None or current_score == None:
        print("No optimal landing point found in region", i)
        continue
    print("Optimal landing point in region", i, "is", current_point, "with score", current_score)
    
    if current_score > global_best_score:
        global_best_score = current_score
        global_best_point = current_point
    
print("Global best landing point is", global_best_point, "with score", global_best_score)


# 3D terrain considering density
x, y = np.meshgrid(np.arange(0, x_num, 1), np.arange(0, y_num, 1))
x_filtered = x[density == 1]
y_filtered = y[density == 1]
z_filtered = z[density == 1]
ax = fig.add_subplot(133, projection='3d')
ax.plot_surface(x, y, z, cmap="terrain", edgecolor='none', alpha=0.3)
ax.scatter(x_filtered, y_filtered, z_filtered, c='b', marker='o', s = 3, alpha=0.45)
ax.scatter(global_best_point[0], global_best_point[1], global_best_point[2], c='r', marker='o', s = 10)
ax.set_title("3D Terrain with Density = 1 Points")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Height")
plt.show()

# Reference Landing Point
reference_i = global_best_point[0]
reference_j = global_best_point[1]
reference = np.array([x_coordinates[reference_j, reference_i], y_coordinates[reference_j, reference_i], z[reference_j, reference_i]])
print("Reference landing point is", reference)

# save the reference point to csv
np.savetxt("reference_point.csv", reference, delimiter=",")