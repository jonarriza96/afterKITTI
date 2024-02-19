import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation

from afterkitti.utils import plot_frames, axis_equal, get_afterkitti_path


def calculate_trajectory(current_frame, imu_yaw, gps_v):
    car_traj = []
    for frame in range(current_frame):

        if frame > 0:
            displacement = -0.1 * np.linalg.norm(gps_v[frame])
            yaw_change = -(imu_yaw[frame] - imu_yaw[frame - 1])
            for i in range(len(car_traj)):
                x0, y0 = car_traj[i]
                x1 = x0 * np.cos(yaw_change) + y0 * np.sin(yaw_change) + displacement
                y1 = -x0 * np.sin(yaw_change) + y0 * np.cos(yaw_change)
                car_traj[i] = np.array([x1, y1])
        car_traj += [np.array([0, 0])]

    car_traj = np.squeeze(car_traj)

    return car_traj


def calculate_future_trajectory(current_frame, imu_yaw, gps_v):
    car_traj = []

    for frame in range(current_frame, len(imu_yaw) - 1):

        if frame > current_frame:
            displacement = 0.1 * np.linalg.norm(gps_v[frame])
            yaw_change = imu_yaw[frame] - imu_yaw[frame - 1]
            for i in range(len(car_traj)):
                x0, y0 = car_traj[i]
                x1 = x0 * np.cos(yaw_change) + y0 * np.sin(yaw_change) + displacement
                y1 = -x0 * np.sin(yaw_change) + y0 * np.cos(yaw_change)
                car_traj[i] = np.array([x1, y1])
        car_traj += [np.array([0, 0])]

    car_traj = np.squeeze(car_traj)

    return car_traj


def plotkitti_2d(frame, pcl_sample, car_p_sample, car_R_sample, imu_yaw, gps_v, time):
    frame = frame + 2

    # ax = plt.figure().add_subplot(111, projection="3d")
    # ax.clear()
    plt.clf()
    # ax = plt.gca(projection="3d")

    fig = plt.gcf()
    ax = fig.gca()  # add_subplot(111, projection="3d")
    ax.scatter(
        pcl_sample[:, 0],
        pcl_sample[:, 1],
        # pcl_sample[:, 2],
        c=-(pcl_sample[:, 0] ** 2 + pcl_sample[:, 1] ** 2 + pcl_sample[:, 2] ** 2),
        cmap="turbo",
        alpha=0.25,
        marker=".",
    )

    plot_frames(
        car_p_sample[None, :],
        car_R_sample[None, :, 0],
        car_R_sample[None, :, 1],
        car_R_sample[None, :, 2],
        ax=ax,
        scale=3,
        planar=True,
    )
    # ax = axis_equal(pcl_sample[:, 0], pcl_sample[:, 1], pcl_sample[:, 2], ax)
    ax.set_aspect("equal")

    car_traj = calculate_trajectory(frame, imu_yaw, gps_v)
    car_traj = np.hstack([car_traj, np.ones((car_traj.shape[0], 1)) * car_p_sample[2]])
    ax.plot(car_traj[:-4, 0], car_traj[:-4, 1], "-m", alpha=0.7)

    car_traj_f = calculate_future_trajectory(frame, imu_yaw, gps_v)
    car_traj_f = np.hstack(
        [car_traj_f, np.ones((car_traj_f.shape[0], 1)) * car_p_sample[2]]
    )
    # ax.plot(car_traj_f[:, 0], car_traj_f[:, 1], car_traj_f[:, 2], "-m")

    plt.title("Frame: " + str(frame) + ", Time: " + str(time) + "s")


def plotkitti_3d(frame, pcl_sample, car_p_sample, car_R_sample, imu_yaw, gps_v, time):
    frame = frame + 2

    # ax = plt.figure().add_subplot(111, projection="3d")
    # ax.clear()
    plt.clf()
    # ax = plt.gca(projection="3d")

    fig = plt.gcf()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        pcl_sample[:, 0],
        pcl_sample[:, 1],
        pcl_sample[:, 2],
        # c=pcl_sample[:, 2],
        c=-(pcl_sample[:, 0] ** 2 + pcl_sample[:, 1] ** 2),
        cmap="turbo",
        alpha=0.25,
        marker=".",
    )

    plot_frames(
        car_p_sample[None, :],
        car_R_sample[None, :, 0],
        car_R_sample[None, :, 1],
        car_R_sample[None, :, 2],
        ax=ax,
        scale=3,
    )
    ax = axis_equal(pcl_sample[:, 0], pcl_sample[:, 1], pcl_sample[:, 2], ax)

    car_traj = calculate_trajectory(frame, imu_yaw, gps_v)
    car_traj = np.hstack([car_traj, np.ones((car_traj.shape[0], 1)) * car_p_sample[2]])
    ax.plot(car_traj[:-4, 0], car_traj[:-4, 1], car_traj[:-4, 2], "-m", alpha=0.7)

    car_traj_f = calculate_future_trajectory(frame, imu_yaw, gps_v)
    car_traj_f = np.hstack(
        [car_traj_f, np.ones((car_traj_f.shape[0], 1)) * car_p_sample[2]]
    )
    # ax.plot(car_traj_f[:, 0], car_traj_f[:, 1], car_traj_f[:, 2], "-m")

    plt.title("Frame: " + str(frame) + ", Time: " + str(time) + "s")


file_name = "kitti_2011_09_26_drive_0005_synced"
afterKITTI_path = get_afterkitti_path()

# ---------------------------------------------------------------------------- #
#                                  Import data                                 #
# ---------------------------------------------------------------------------- #s
with open(afterKITTI_path + "/data/pickles/" + file_name + ".pkl", "rb") as f:
    data = pickle.load(f)

car_p = data["car"]["p"]
car_q = data["car"]["q"]
pcl_t = data["pcl"]["t"]
pcl = data["pcl"]["pcl"]
imu_yaw = data["imu"]["yaw"]
gps_v = data["gps"]["v"]

# ---------------------------------------------------------------------------- #
#                                   Visualize                                  #
# ---------------------------------------------------------------------------- #
fig_2d = plt.figure()
z_threshold = -0.5
frames = len(pcl_t)
animation_2d = FuncAnimation(
    fig_2d,
    lambda frame_num: plotkitti_2d(
        frame=frame_num,
        pcl_sample=pcl[frame_num][pcl[frame_num][:, 2] > z_threshold],
        car_p_sample=car_p[frame_num],
        car_R_sample=Rotation.from_quat(car_q[frame_num]).as_matrix(),
        imu_yaw=imu_yaw,
        gps_v=gps_v,
        time=round(frame_num / frames * pcl_t[-1], 3),
    ),
    frames=frames - 4,
)
plt.show()


fig_3d = plt.figure()
animation_3d = FuncAnimation(
    fig_3d,
    lambda frame_num: plotkitti_3d(
        frame=frame_num,
        pcl_sample=pcl[frame_num][pcl[frame_num][:, 2] > z_threshold],
        car_p_sample=car_p[frame_num],
        car_R_sample=Rotation.from_quat(car_q[frame_num]).as_matrix(),
        imu_yaw=imu_yaw,
        gps_v=gps_v,
        time=round(frame_num / frames * pcl_t[-1], 3),
    ),
    frames=frames - 4,
)

print("Saved animations in:", afterKITTI_path + "docs/ ...")
# animation_2d.save(
#     afterKITTI_path + "docs/" + "kitti_2d.gif",
#     writer="ffmpeg",
# )
# animation_3d.save(
#     afterKITTI_path + "docs/" + "kitti_3d.gif",
#     writer="ffmpeg",
# )
print("Done.")
plt.show()
