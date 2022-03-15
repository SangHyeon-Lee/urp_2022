from tokenize import PlainToken
import numpy as np
import matplotlib.pyplot as plt

import util
import open3d as o3d
from tqdm import tqdm




# normal vector of face
test_obj = util.ObjFile()

print(test_obj.f[100][:,0])

v1_idx, v2_idx, v3_idx = test_obj.f[1][:, 0]

v1 = test_obj.v[v1_idx]
v2 = test_obj.v[v2_idx]
v3 = test_obj.v[v3_idx]

print(v1, v2, v3)




A = np.array([v1, v2, v3], dtype='float64')
b = np.array([-1, -1, -1])


print("start")

x = np.linalg.solve(A, b)

print(np.sum(x*v2))
print(np.sum(x*v3))

print(x)


test_obj.get_face_plane_coeff(1)


# SYYI commented out
#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#ax.view_init(-90, 90)


print("face sampling start")
points = np.zeros((0,3)) # SYYI appended
colors = np.zeros((0,3)) # SYYI appended
#for i in range(1, len(test_obj.f)): # SYYI commented out
for i in tqdm(range(1, len(test_obj.f))): # SYYI appended
    x, c = test_obj.get_points_on_face(i)
    #print(x)
    if len(x) == 0:
        continue
    X = x[:, 0]
    Y = x[:, 1]
    Z = x[:, 2]
    
    # SYYI appended #
    points = np.concatenate([points, x])
    colors = np.concatenate([colors, c])
    #print(c)
    #ax.scatter(X, Y, Z, c = c, alpha=0.6, linewidths=0.8)
    # SYYI end #
    

    #vt1, vt2, vt3 = test_obj.get_face_vt(i)
# SYYI appended #
print("points.shape=%s" % (points.shape,))
print("points[0]=%s" % points[0])
print("colors.shape=%s" % (colors.shape,))
print("colors[0]=%s" % colors[0])

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors*2)
pcd.normals = o3d.utility.Vector3dVector(points / np.expand_dims(np.linalg.norm(points, axis=1), axis=1))

o3d.visualization.draw_geometries([pcd])
# SYYI end

#plt.imshow(test_obj.texture)

#plt.show()