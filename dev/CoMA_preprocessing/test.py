from tokenize import PlainToken
import numpy as np
import matplotlib.pyplot as plt

import util





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



fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.view_init(-90, 90)


print("face sampling start")
for i in range(1, len(test_obj.f)):
    x, c = test_obj.get_points_on_face(i)
    #print(x)
    if len(x) == 0:
        continue
    X = x[:, 0]
    Y = x[:, 1]
    Z = x[:, 2]
    
    #print(c)
    ax.scatter(X, Y, Z, c = c, alpha=0.3)

    #vt1, vt2, vt3 = test_obj.get_face_vt(i)



#plt.imshow(test_obj.texture)

plt.show()