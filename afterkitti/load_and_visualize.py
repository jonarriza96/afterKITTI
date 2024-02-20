import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation

from afterkitti.utils import (
    get_afterkitti_path,
    plotkitti_2d,
    plotkitti_3d,
    save_animation,
)


file_name = "kitti_2011_09_26_drive_0005_synced"

# ---------------------------------------------------------------------------- #
#                                  Import data                                 #
# ---------------------------------------------------------------------------- #s
with open(get_afterkitti_path() + "/data/pickles/" + file_name + ".pkl", "rb") as f:
    data = pickle.load(f)

world_p = data["world"]["p"]
world_q = data["world"]["q"]
car_p = data["car"]["p"]
car_q = data["car"]["q"]
pcl_t = data["pcl"]["t"]
pcl = data["pcl"]["pcl"]
imu_yaw = data["imu"]["yaw"]
gps_v = data["gps"]["v"]


# ---------------------------------------------------------------------------- #
#                                   Visualize                                  #
# ---------------------------------------------------------------------------- #


wq = np.hstack([world_q[:, 1:], world_q[:, 0][:, None]])

z_threshold = -0.5
frames = len(pcl_t)


fig_2d = plt.figure()
animation_2d = FuncAnimation(
    fig_2d,
    lambda frame_num: plotkitti_2d(
        frame=frame_num,
        pcl_sample=(pcl[frame_num][pcl[frame_num][:, 2] > z_threshold])
        @ Rotation.from_quat(wq[frame_num]).as_matrix().T
        + world_p[frame_num],
        car_p_sample=world_p[frame_num],
        car_R_sample=Rotation.from_quat(wq[frame_num]).as_matrix(),
        car_traj=world_p,
        time=round(frame_num / frames * pcl_t[-1], 3),
    ),
    frames=frames - 4,
)
# save_animation(animation_2d, "kitti_2d")
plt.show()


fig_3d = plt.figure()
animation_2d = FuncAnimation(
    fig_3d,
    lambda frame_num: plotkitti_3d(
        frame=frame_num,
        pcl_sample=(pcl[frame_num][pcl[frame_num][:, 2] > z_threshold])
        @ Rotation.from_quat(wq[frame_num]).as_matrix().T
        + world_p[frame_num],
        car_p_sample=world_p[frame_num],
        car_R_sample=Rotation.from_quat(wq[frame_num]).as_matrix(),
        car_traj=world_p,
        time=round(frame_num / frames * pcl_t[-1], 3),
    ),
    frames=frames - 4,
)
# save_animation(animation_2d, "kitti_3d")
plt.show()
