import numpy as np
from matplotlib import image
import matplotlib.pyplot as plt

temp_path = "data/bareteeth.000001.obj"

class ObjFile:
    def read_obj(self):
        print("obj file parsing start...")

        file = open(self.path, 'r')
        lines = file.readlines()

        # data lists (with padding)
        v = [[None, None, None]]
        vn = [[None, None, None]]
        vt = [[None, None]]
        f = [
                [[None, None, None],[None, None, None],[None, None, None]]
            ]   

        
        # parse obj file
        for line in lines:
            components = line.split(" ")

            if len(components) <= 1:
                continue
            
            cat = components[0]

            if cat == 'v':
                x = float(components[1])
                y = float(components[2])
                z = float(components[3][:-1])
                v.append([x, y, z])

            elif cat == 'vt':
                x = float(components[1])
                y = float(components[2][:-1])
                vt.append([x, y])
            elif cat == 'vn':
                x = float(components[1])
                y = float(components[2])
                z = float(components[3][:-1])
                vn.append([x, y, z])
            elif cat == 'f':
                v1 = components[1]
                v2 = components[2]
                v3 = components[3][:-1]

                v1, vt1, vn1 = v1.split("/")
                v2, vt2, vn2 = v2.split("/")
                v3, vt3, vn3 = v3.split("/")

                f.append([
                        [int(v1), int(vt1), int(vn1)],
                        [int(v2), int(vt2), int(vn2)],
                        [int(v3), int(vt3), int(vn3)]
                    ])

        
        file.close()

        v = np.array(v, dtype='float64')
        vn = np.array(vn, dtype='float64')
        vt = np.array(vt, dtype='float64')
        f = np.array(f)

        print("complete!")

        return v, vn, vt, f


        
    
    def __init__(self, obj_path=temp_path):
        self.path = obj_path
        self.v, self.vn, self.vt, self.f = self.read_obj()


        self.texture = image.imread(obj_path[:-3] + "png")
    
    def get_face_vertex(self, face_idx):
        v1_idx, v2_idx, v3_idx = self.f[face_idx][:,0]

        v1, v2, v3 = self.v[v1_idx], self.v[v2_idx], self.v[v3_idx]

        return v1, v2, v3
    
    def get_face_vt(self, face_idx):
        vt1_idx, vt2_idx, vt3_idx = self.f[face_idx][:,1]
        vt1, vt2, vt3 = self.vt[vt1_idx], self.vt[vt2_idx], self.vt[vt3_idx]

        return vt1, vt2, vt3


    def get_face_plane_coeff(self, face_idx):
        assert face_idx != 0, 'invalid index'

        v1, v2, v3 = self.get_face_vertex(face_idx)


        # solve linear eqn
        A = np.array([v1, v2, v3], dtype='float64')
        b = np.array([-1, -1, -1])

        # plane equation
        a, b, c = np.linalg.solve(A, b)
        d = 1

        # ax + by + cz + d = 0
        return a, b, c, d
    
    def get_points_on_face(self, face_idx):
        
        # vertices of face
        v1, v2, v3 = self.get_face_vertex(face_idx)

        # face edges
        a = v2 - v1
        b = v3 - v1
        c = v3 - v2
        
        # norm of a
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        # orthogonal component of b wrt a
        b_orth = b - a*np.dot(a, b)/norm_a/norm_b
        norm_b_orth = np.linalg.norm(b_orth)


        # number of sample points (fix)
        N = 10 # fix
        N_a = int((norm_a/(norm_a + norm_b_orth))*N)
        N_b = N - N_a


        # parameterization
        alphas = np.linspace(0, 1, N_a)
        betas = np.linspace(0, 1, N_b)



        # texture space
        vt1, vt2, vt3 = self.get_face_vt(face_idx)

        at = vt2 - vt1
        bt = vt3 - vt1

        norm_at = np.linalg.norm(a)
        norm_bt = np.linalg.norm(b)

        #bt_orth = bt - at * np.dot(at, bt)/norm_at/norm_bt # SYYI commented out
        bt_orth = bt - at * np.dot(a, b)/norm_a/norm_b # SYYI appended
        norm_bt_orth = np.linalg.norm(bt_orth)
                







        points = []
        valid_params = []
        colors = []
        for alpha in alphas:
            for beta in betas:
                point_rel = alpha * a + beta * b_orth
                point = v1 + point_rel
                
                theta_max1 = np.dot(b, b_orth)/norm_b/norm_b_orth
                theta_max2 = np.dot(b - a, b_orth)/np.linalg.norm(b - a)/norm_b_orth

                

                if np.linalg.norm(point_rel-a) < 0.0001:
                    #print(a, point_rel)
                    if np.linalg.norm(point_rel -  a) == 0:
                        continue
                        pass#print("Gotcha!")
                theta1 = np.dot(point_rel, b_orth)/np.linalg.norm(point_rel)/norm_b_orth
                theta2 = np.dot(point_rel - a, b_orth)/np.linalg.norm(point_rel - a)/norm_b_orth

                #print(theta1, theta2, theta_max1, theta_max2)
                

                # color information
                img_h, img_w, _ = self.texture.shape

                point_rel_t = alpha * at + beta * bt_orth
                point_t = vt1 + point_rel_t
                #print(point_t)
                #print("color", self.texture[int(point_t[0] * w)][int(point_t[1] * h)])

                img_dim = np.array((img_w, img_h))
                point_t = point_t * img_dim

                #print(point_t, vt1, img_dim)


                if theta1 < theta_max1 and theta2 < theta_max2:
                    points.append(point)
                    colors.append(self.texture[img_h-1-int(point_t[1])][int(point_t[0])]) # SYYI modified
                #else:
                #    points.append(point)
        
        #print(points)

        return np.array(points), np.array(colors)