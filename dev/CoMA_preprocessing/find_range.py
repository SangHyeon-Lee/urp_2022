from cgi import test
import os
from re import sub
import util
import numpy as np
import csv


coma_dir = "/media/vclab/extSSD/FaceTalk_170915_00223_TA"
save_dir = "/media/vclab/extSSD/pcl_coma/"


x_min = 987654321
x_max = -987654321
y_min = 987654321
y_max = -987654321
z_min = 987654321
z_max = -987654321



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
        #print(dirName)
        testcase_dir = dirName + '/' + file
        obj = util.ObjFile(testcase_dir)

        x = obj.v[:, 0]
        x = x[~np.isnan(x)]
        y = obj.v[:, 1]
        y = y[~np.isnan(y)]
        z = obj.v[:, 2]
        z = z[~np.isnan(z)]

        x_max = max(x_max, max(x))
        y_max = max(y_max, max(y))
        z_max = max(z_max, max(z))

        x_min = min(x_min, min(x))
        y_min = min(y_min, min(y))
        z_min = min(z_min, min(z))

    print(dirName)
    print(x_max, x_min, y_max, y_min, z_max, z_min)
    
    f = open('range_out.csv', 'w')
    writer = csv.writer(f)
    writer.writerow((x_max, x_min, y_max, y_min, z_max, z_min))
    f.close()






    



    
    