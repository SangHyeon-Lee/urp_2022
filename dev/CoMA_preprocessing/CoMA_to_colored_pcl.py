import enum
import os
import util
import numpy as np
import csv
import open3d as o3d
import laspy as lp
from tqdm import tqdm


# directory setup
coma_dir = "/media/vclab/extSSD/FaceTalk_170915_00223_TA"
save_dir = "/media/vclab/extSSD/pcl_coma/"


# dataset out range
x_min, x_max = -200, 200
y_min, y_max = -200, 200
z_min, z_max = -200, 200

linspace_x = np.linspace(x_min, x_max, 51)
linspace_y = np.linspace(y_min, y_max, 51)
linspace_z = np.linspace(z_min, z_max, 51)




# CoMA directory iteration
for dirName, subdirList, fileList in os.walk(coma_dir):
    #print(fileList[:3], dirName.split('/')[-1])

    testcase_name = dirName.split('/')[-1]

    if fileList == []:
        continue
    

    testcase_save_dir = save_dir + testcase_name
    if not os.path.exists(testcase_save_dir):
        os.makedirs(testcase_save_dir)

    for file in fileList:
        extension = file.split('.')[-1]

        if extension != 'obj':
            continue
        

        # read .obj file
        testcase_dir = dirName + '/' + file
        obj = util.ObjFile(testcase_dir)

        points = np.zeros((0, 3))
        colors = np.zeros((0, 3))
        for i in tqdm(range(1, len(obj.f), 200)):
            x, c = obj.get_points_on_face(i)


            point_list = []
            color_list = []
            for j in range(len(x)):
                if np.isnan(x[j, 0]) or np.isnan(x[j, 1]) or np.isnan(x[j, 2]):
                    continue
                
                points = np.concatenate ([points, np.array([x[j]])])
                colors = np.concatenate ([colors, np.array([c[j]])])

            #x = x[ ~np.isnan(x[:, 0]) and ~np.isnan(x[:, 1]) and ~np.isnan(x[:, 2])]
            #c = c[ ~np.isnan(x[:, 0]) and ~np.isnan(x[:, 1]) and ~np.isnan(x[:, 2])]

            #points = np.concatenate ([points, x])
            #colors = np.concatenate ([colors, c])


        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors * 2)
        pcd.normals = o3d.utility.Vector3dVector(points / np.expand_dims (np.linalg.norm(points, axis = 1), axis = 1))

        #o3d.visualization.draw_geometries([pcd])


        # voxelization
        v_size = round (max(pcd.get_max_bound() - pcd.get_min_bound())* 0.005, 4)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=v_size)

        #o3d.visualization.draw_geometries([voxel_grid])
        voxel = voxel_grid.get_voxels()
        voxel_pos = np.array([vox.grid_index for vox in voxel])
        voxel_color = np.array([vox.color for vox in voxel])


        print(voxel_pos[:10])
        break


    break



    print(dirName)
    print(x_max, x_min, y_max, y_min, z_max, z_min)
