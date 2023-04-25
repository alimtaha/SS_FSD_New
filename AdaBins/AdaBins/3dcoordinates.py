import numpy as np
from path import Path
import torch
from matplotlib import projections, pyplot as plt
from PIL import Image
import json
from mpl_toolkits.mplot3d import proj3d
import pptk
import open3d
import os

dirs = '/home/extraspace/Datasets/Datasets/cityscapes/leftImg8bit_trainvaltest_(blurred)/leftImg8bit/'
str = '/home/extraspace/Datasets/Datasets/cityscapes/city/depth_adabins'
str2 = '/home/extraspace/Datasets/Datasets/cityscapes/leftImg8bit_trainvaltest_(blurred)/leftImg8bit/val/frankfurt/'
str3 = '/home/extraspace/Datasets/Datasets/cityscapes/depth/multi_new_depth_inf/val/frankfurt/'
str4 = '/home/taha_a@WMGDS.WMG.WARWICK.AC.UK/Documents/ss_fsd/NeWCRFs/models/result_newcrfs_cityscapes/raw/leftImg8bit_'
label = '/home/extraspace/Datasets/Datasets/cityscapes/city/segmentation/val/frankfurt_000000_000294_gtFine.png'


label = np.array(Image.open(label), dtype=np.int)
ego_mask = np.array(label > 250).astype(int)
ego_mask[:600, ...] = 0
ego_mask = ego_mask.reshape(-1)

img_list = []
#img_list = ['frankfurt_000000_000294.png', 'frankfurt_000000_000576.png',
#            'frankfurt_000000_001016.png', 'frankfurt_000000_001236.png']


#generating list of all dirs
for i in os.listdir(dirs):
    if i in ['train','val']:
        for m in os.listdir(os.path.join(dirs, i)):
            for l in os.listdir(os.path.join(dirs, i, m)):
                img_list.append(os.path.join(dirs, i, m, l).replace('.jpg', '.png'))

i = 0

#generating manual depth maps - change for different thresholds
for thresh in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    save_path_root = f'/home/extraspace/Datasets/Datasets/cityscapes/city/segmentation_weak_{thresh}'

    for img in img_list:
        
        i += 1
        print(i)

        x = np.linspace(0, 2047, 2048)
        y = np.linspace(0, 1023, 1024)
        xz, yz = np.meshgrid(x, y)
        pixel_pitch = 2.2e-6
        
        subdir = ('/').join(img.split('/')[-3:-1])
        save_root_subdir = os.path.join(save_path_root, subdir)
        
        if not os.path.exists(save_root_subdir):
            os.makedirs(save_root_subdir)

        path_ = img.split('/')[-1]

        #print(path_)
        #print(subdir)
        #print(save_root_subdir)
        
        depth_path = os.path.join(str, path_)
        img = np.array(Image.open(img.replace('.png', '.jpg')), dtype=np.float32)
        depth = np.asarray(Image.open(depth_path), np.float32)
        depth = depth / 256.0
        camera = (os.path.join('/home/extraspace/Datasets/Datasets/cityscapes/camera/', subdir, path_)).replace('.png', '_camera.json')

        with open(camera, 'r') as f:
            camera_params = json.loads(f.read())
            ox = camera_params['intrinsic']['u0']
            oy = camera_params['intrinsic']['v0']
            fx = camera_params['intrinsic']['fx']
            fy = camera_params['intrinsic']['fy']
            z = camera_params['extrinsic']['z']
            x = camera_params['extrinsic']['x']
            y = camera_params['extrinsic']['y']
            roll = camera_params['extrinsic']['roll']
            pitch = camera_params['extrinsic']['pitch']
            yaw = camera_params['extrinsic']['yaw']

            s_p = np.sin(pitch)
            s_y = np.sin(yaw)
            s_r = np.sin(roll)
            c_p = np.cos(pitch)
            c_y = np.cos(yaw)
            c_r = np.cos(roll)

            xz = xz.reshape(-1)
            yz = yz.reshape(-1)
            # in metres, need to convert to image coordinates
            depth_reshape = depth.reshape(-1)
            zz = depth_reshape / pixel_pitch  #
            xz = (xz-ox)*zz/fx
            yz = (yz-oy)*zz/fy


            # convert pixels to metres
            zz *= pixel_pitch
            xz *= -pixel_pitch  # negative one to allow y-axis to point left not right
            yz *= -pixel_pitch  # negative one to allow y-axis to point upwards not downwards
            # convert z to ground plane coordinates - 1.18
            # axis offset from bottom of car

            #rotation and translation
            zz = c_y*c_p*zz + (c_y*s_p*s_r-s_y*c_r)*xz + (c_y*s_p*c_r+s_y+s_r)*yz + x
            #changed minus in middle term to plus
            xz = s_y*c_p*zz + (s_y*s_p*s_r+c_y*c_r)*xz + (s_y*s_p*c_r-c_y*s_r)*yz + y
            yz = -s_p*zz + c_p*s_r*xz + c_p*c_r*yz + z

            #print(xz.shape, yz.shape, depth_coordinates.shape)

            non_ground_mask = (yz > thresh).astype(int)
            non_ground_mask = np.logical_or(non_ground_mask, np.abs(ego_mask))[..., np.newaxis]

            #labels = np.zeros_like(yz, dtype = int)

            labels = non_ground_mask.reshape(1024, 2048).astype(np.uint8)

            save_path = os.path.join(save_root_subdir, path_)

            im_save = Image.fromarray(labels)
            im_save.save(save_path)

    #fig = plt.figure(figsize=(10, 10))
    #fig.add_subplot(2, 1, 1)
    #plt.imshow((labels).astype(int))
    #fig.add_subplot(2, 1, 2)
    #plt.imshow(colors)
    #plt.show()




#pt_cloud = np.stack((xz, yz, zz), axis=1)
        # print(pt_cloud.shape)
        #colors = np.asarray(img, np.float32)/255.0
        #colors = colors.reshape(-1, 3)

        # 20 cm estimation for ground
        

        #road_color = (np.array((0, 0, 1))[..., np.newaxis]).T
        #road_color = np.repeat(road_color, 2097152, axis=0)
        # print(road_color.shape)
        #colors = colors*(1-ground_mask) + road_color*ground_mask







# print(colors.shape)

#pcd = open3d.geometry.PointCloud()
#pcd.points = open3d.utility.Vector3dVector(pt_cloud)
#pcd.colors = open3d.utility.Vector3dVector(colors)
#pcd.normals = open3d.utility.Vector3dVector(normals)
# open3d.visualization.draw_geometries([pcd])

#vis = open3d.visualization.Visualizer()
# vis.create_window()
# vis.add_geometry([pcd])
#ctr = vis.get_view_control()
# ctr.change_field_of_view(step=90)
# vis.run()

#xyz = pptk.rand(100, 3)
# pptk.viewer(xyz)
# pptk.viewer(pt_cloud[:100,...])

# print(depth_reshape[1050623])
#depth_rereshape = depth_reshape.reshape((1024, 2048))
# print(depth_rereshape[512,2047])

# plt.imshow(img)
# plt.show()
#plt.imshow(depth, cmap='magma_r')
# plt.colorbar()
# plt.show()

#fig = plt.figure(figsize=(10,10))
#ax = fig.add_subplot(111, projection='3d')
# ax.scatter(xz,yz,depth_coordinates)
# plt.show()

#ground_mask_reshape = (yz < 0.2).astype(int).reshape(1024, 2048)
# ground_mask_reshape = np.stack(
#    (ground_mask_reshape, ground_mask_reshape, ground_mask_reshape), axis=2)
#colours = np.repeat(np.array([0, 0, 255]), 2097152).reshape(1024, 2048, 3)

#colors *= 255.0
#colors = colors.astype(int).reshape(1024, 2048, 3)  # .transpose(1,2,0)

#img_viz = np.array(Image.open((str2+img_list[i]).replace('.png', '.jpg')))
#img_viz = img_viz * (1 - ground_mask_reshape) + colours * ground_mask_reshape



# inverse projection
# zz = c_y*c_p*zz + s_y*c_p*xz - s_p*yz - ( \
#    c_y*c_p*x + (c_y*s_p*s_r-s_y*c_r)*y + (c_y*s_p*c_r+s_y+s_r)*z)
# xz = (c_y*s_p*s_r-s_y*c_r)*zz + (s_y*s_p*s_r-c_y*c_r)*xz + c_p*s_r*yz -( \
#    s_y*c_p*x + (s_y*s_p*s_r-c_y*c_r)*y + (s_y*s_p*c_r-c_y*s_r)*z )
# yz = (c_y*s_p*c_r+s_y*s_r)*zz + (s_y*s_p*c_r-c_y*s_r)*xz + c_p*c_r*yz -( \
#    -s_p*x + c_p*s_r*y + c_p*c_r*z)
