# %%
# %matplotlib tk
import numpy as np
import pickle

import rosbag
import sensor_msgs.point_cloud2 as pc2

from scipy.spatial.transform import Rotation
from afterkitti.utils import get_afterkitti_path


file_name = "kitti_2011_09_26_drive_0005_synced"

# ---------------------------------------------------------------------------- #
#                                  Import data                                 #
# ---------------------------------------------------------------------------- #
print("Converting rosbag to python pickle ...")
afterKITTI_path = get_afterkitti_path()
bag_file = afterKITTI_path + "/data/rosbags/" + file_name + ".bag"
bag = rosbag.Bag(bag_file)

# -------------------------------- Point cloud ------------------------------- #
pcl = []
pcl_t = []
for topic, msg, t in bag.read_messages(topics=["/kitti/velo/pointcloud"]):
    gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    pcl += [np.array(list(gen))]
    pcl_t += [t.to_sec()]
pcl_t = np.squeeze(pcl_t) - pcl_t[0]

# --------------------------------- Car pose --------------------------------- #
car_p = []
car_q = []
car_t = []
for topic, msg, t in bag.read_messages(topics=["/tf_static"]):
    for tr in msg.transforms:
        if tr.header.frame_id == "imu_link":
            car_p += [
                np.array(
                    [
                        tr.transform.translation.x,
                        tr.transform.translation.y,
                        tr.transform.translation.z,
                    ]
                ),
            ]
            car_q += [
                np.array(
                    [
                        tr.transform.rotation.w,
                        tr.transform.rotation.x,
                        tr.transform.rotation.y,
                        tr.transform.rotation.z,
                    ]
                ),
            ]
            car_t += [t.to_sec()]
world_p = []
world_q = []
world_t = []
for topic, msg, t in bag.read_messages(topics=["/tf"]):
    for tr in msg.transforms:
        if tr.header.frame_id == "world":
            world_p += [
                np.array(
                    [
                        tr.transform.translation.x,
                        tr.transform.translation.y,
                        tr.transform.translation.z,
                    ]
                ),
            ]
            world_q += [
                np.array(
                    [
                        tr.transform.rotation.w,
                        tr.transform.rotation.x,
                        tr.transform.rotation.y,
                        tr.transform.rotation.z,
                    ]
                ),
            ]
            world_t += [t.to_sec()]


car_p = np.squeeze(car_p)
car_q = np.squeeze(car_q)
car_t = np.squeeze(car_t) - car_t[0]

world_p = np.squeeze(world_p)
world_q = np.squeeze(world_q)
world_t = np.squeeze(world_t) - world_t[0]

# --------------------------------- GPS data --------------------------------- #
gps_v = []
for topic, msg, t in bag.read_messages(topics=["/kitti/oxts/gps/vel"]):
    gps_v += [np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])]
gps_v = np.squeeze(gps_v)

# --------------------------------- IMU data --------------------------------- #
imu_yaw = []
for topic, msg, t in bag.read_messages(topics=["/kitti/oxts/imu"]):
    q_imu = np.array(
        [msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z]
    )
    euler = Rotation.from_quat(q_imu).as_euler("zyx")
    imu_yaw += [euler[-1]]
imu_yaw = np.squeeze(imu_yaw)

bag.close()
print("Done.")


# ---------------------------------------------------------------------------- #
#                                Save to pickle                                #
# ---------------------------------------------------------------------------- #
data = {
    "world": {"p": world_p, "q": world_q, "t": world_t},
    "car": {"p": car_p, "q": car_q, "t": car_t},
    "pcl": {"pcl": pcl, "t": pcl_t},
    "imu": {"yaw": imu_yaw},
    "gps": {"v": gps_v},
}
with open(afterKITTI_path + "/data/pickles/" + file_name + ".pkl", "wb") as f:
    pickle.dump(data, f)
