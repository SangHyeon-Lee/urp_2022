import enum
import os
import util
import numpy as np
import csv
import open3d as o3d
from tqdm import tqdm


# directory setup
coma_dir = "/media/vclab/extSSD/FaceTalk_170915_00223_TA"
save_dir = "/media/vclab/extSSD/pcl_coma/"


# dataset out range
x_min, x_max = -200, 200
y_min, y_max = -200, 200
z_min, z_max = -150, 160

linspace_x = np.linspace(x_min, x_max, 50)
linspace_y = np.linspace(y_min, y_max, 50)
linspace_z = np.linspace(z_min, z_max, 50)




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
        for i in tqdm(range(1, len(obj.f), 10)):
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




        occupancy_array = []
        point_array = []
        color_array = []
        
        # iterate voxel
        for i in tqdm(range(len(linspace_x) - 1)):
            for j in range(len(linspace_y) - 1):
                for k in range(len(linspace_z) - 1):
                    x_left, x_right = linspace_x[i], linspace_x[i + 1]
                    y_left, y_right = linspace_y[i], linspace_y[i + 1]
                    z_left, z_right = linspace_z[i], linspace_z[i + 1]

                    x_mid = (x_left + x_right)/2
                    y_mid = (y_left + y_right)/2
                    z_mid = (z_left + z_right)/2

                    
                    target_point_idx_list = []
                    for l in range(len(points)):
                        cond_x =  points[l, 0] > x_left and points[l, 0] < x_right
                        cond_y =  points[l, 1] > y_left and points[l, 1] < y_right
                        cond_z =  points[l, 2] > z_left and points[l, 2] < z_right

                        if cond_x and cond_y and cond_z:
                            # find point in boxel
                            target_point_idx_list.append(l)
                    

                    if len(target_point_idx_list) == 0:
                        # there's no point in this voxel
                        occupancy_array.append(0)
                        color_array.append(None)
                        point_array.append(np.array([x_mid, y_mid, z_mid]))
                    else:
                        print(len(target_point_idx_list))
                        pass
        






        break


    break



    print(dirName)
    print(x_max, x_min, y_max, y_min, z_max, z_min)
