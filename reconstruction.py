import numpy as np
import pyrender

K = np.array([[1268.6689941574389, 0.0, 915.0310459719078],
              [0.0, 1321.3619862952573, 544.5298812488835],
              [0.0, 0.0, 1.0]])

R = np.array([[0.8891977,  0.00252625, 0.45751612],
              [-0.0647056, 0.99062813, 0.12028756],
              [-0.45292446,-0.13656328,0.88102775]])
t = np.array([-1.38004893, -0.02757486, 1.22584611])  # world->cam

# 1) Intrinsics for pyrender
fx, fy = K[0,0], K[1,1]
cx, cy = K[0,2], K[1,2]
cam = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)

# 2) camera_pose: camera -> world
cam_pose = np.eye(4)
cam_pose[:3,:3] = R.T
cam_pose[:3, 3] = -R.T @ t

scene = pyrender.Scene()
scene.add(cam, pose=cam_pose)
# 再把你的 SMPL mesh / 点云加进去，就可以渲染了
