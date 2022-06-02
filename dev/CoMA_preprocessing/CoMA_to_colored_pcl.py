from cgi import test
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


linspace_x = np.linspace(x_min, x_max, 101)
linspace_y = np.linspace(y_min, y_max, 101)
linspace_z = np.linspace(z_min, z_max, 101)




# CoMA directory iteration
for dirName, subdirList, fileList in os.walk(coma_dir):
    #print(fileList[:3], dirName.split('/')[-1])

    testcase_name = dirName.split('/')[-1]

    if fileList == []:
        continue
    

    testcase_save_dir = save_dir + testcase_name
    
    if not os.path.exists(testcase_save_dir):
        os.makedirs(testcase_save_dir)

    for file in sorted(fileList):
        extension = file.split('.')[-1]

        if extension != 'obj':
            continue
        
        
        #print(testcase_save_dir, file)
        output_save_dir = testcase_save_dir + "/" + file[:-4] + ".npz"
        print("processing: ", output_save_dir)

        # read .obj file
        testcase_dir = dirName + '/' + file
        obj = util.ObjFile(testcase_dir)

        points = np.zeros((0, 3))
        colors = np.zeros((0, 3))
        for i in tqdm(range(1, len(obj.f), 10)):
            x, c = obj.get_points_on_face(i)


            point_list = []
            color_list = []
            for j in range(0, len(x), 2):
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
        voxel_min_bound = np.array([[x_min],[y_min],[z_min]])
        voxel_max_bound = np.array([[x_max],[y_max],[z_max]])
        v_size = round (max(pcd.get_max_bound() - pcd.get_min_bound())* 0.005, 4)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(pcd, 4, voxel_min_bound, voxel_max_bound)


        #o3d.visualization.draw_geometries([voxel_grid])
        voxel = voxel_grid.get_voxels()
        voxel_pos = np.array([vox.grid_index for vox in voxel])
        voxel_color = np.array([vox.color for vox in voxel])


        #print(voxel_pos[:30])
        #print(voxel_grid)
        abs_pos = (voxel_grid.origin + 4 * voxel_pos)
        #print(abs_pos[:10])
        #print(min(abs_pos[:, 2]))

        #print(np.shape(abs_pos))
        #print(voxel_color[:10])

        



        L = len(linspace_x)

        points = np.zeros((L**3, 3))
        occupancies = np.zeros((L**3, 1), dtype=np.int_)
        colors = np.zeros((L**3, 3))

        occ_idx = np.zeros(L**3, dtype=np.uint8)
        idx_table = np.zeros(L**3, dtype=np.uint8) 

        for i, pos in enumerate(abs_pos):

            x_idx = np.where(linspace_x == pos[0])
            y_idx = np.where(linspace_y == pos[1])
            z_idx = np.where(linspace_z == pos[2])

            #print(x_idx, y_idx, z_idx)
            x_idx, y_idx, z_idx = x_idx[0], y_idx[0], z_idx[0]
            idx = x_idx * L * L + y_idx * L + z_idx

            occ_idx[idx] = 1
            idx_table[idx] = i


        for i in tqdm(range(L*L*L)):
            # initial values
            x, y, z = (i // L // L) % L, (i // L) % L, i % L

            assert(i == x * L * L + y * L + z)
            point = np.array([[x, y, z]])
            occupancy = np.array([[0]], dtype=np.int_)
            color = np.array([[0, 0, 0]])

            if occ_idx[i] == 1:
                idx = idx_table[i]
                occupancy = np.array([[1]], dtype=np.int_)
                color = np.array([voxel_color[idx]])

            points[i,:] = point
            occupancies[i,:] = occupancy
            colors[i,:] = color
            #occupancies = np.concatenate([occupancies, occupancy])
            #colors = np.concatenate([colors, color])

        #print(points[:10,:])

        '''
        for x in tqdm(linspace_x):
            for y in linspace_y:
                for z in linspace_z:

                    # initial values
                    point = np.array([[x, y, z]])
                    occupancy = np.array([[0]], dtype=np.int_)
                    color = np.array([[0, 0, 0]])
                    
                    for i, pos in enumerate(abs_pos):
                        if x == pos[0] and y == pos[1] and z == pos[2]:
                            #print(x, y, z)
                            occupancy = np.array([[1]], dtype=np.int_)
                            color = np.array([voxel_color[i]])

                            np.delete(abs_pos, i)
                            break
                    
                    points = np.concatenate([points, point])
                    occupancies = np.concatenate([occupancies, occupancy])
                    colors = np.concatenate([colors, color])
        '''


        #out_dict = {}
        #out_dict['points'] = points
        #out_dict['occupancies'] = occupancies
        #out_dict['colors'] = colors


        np.savez(output_save_dir, points=points, occupancies=occupancies, colors=colors)



    #print(dirName)
    #print(x_max, x_min, y_max, y_min, z_max, z_min)
