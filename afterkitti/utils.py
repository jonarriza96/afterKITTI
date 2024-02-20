import numpy as np
import matplotlib.pyplot as plt
import subprocess


def get_afterkitti_path():
    afterkitti_path = subprocess.run(
        "echo $AFTERKITTI_PATH", shell=True, capture_output=True, text=True
    ).stdout.strip("\n")
    return afterkitti_path


def axis_equal(X, Y, Z, ax=None):
    """
    Sets axis bounds to "equal" according to the limits of X,Y,Z.
    If axes are not given, it generates and labels a 3D figure.

    Args:
        X: Vector of points in coord. x
        Y: Vector of points in coord. y
        Z: Vector of points in coord. z
        ax: Axes to be modified

    Returns:
        ax: Axes with "equal" aspect


    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    max_range = (
        np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() / 2.0
    )
    mid_x = (X.max() + X.min()) * 0.5
    mid_y = (Y.max() + Y.min()) * 0.5
    mid_z = (Z.max() + Z.min()) * 0.5
    ax.set_xlim(mid_x - 1.2 * max_range, mid_x + 1.2 * max_range)
    ax.set_ylim(mid_y - 1.2 * max_range, mid_y + 1.2 * max_range)
    ax.set_zlim(mid_z - 1.2 * max_range, mid_z + 1.2 * max_range)

    return ax


def plot_frames(
    r, e1, e2, e3, interval=0.9, scale=1.0, ax=None, ax_equal=True, planar=False
):
    """
    Plots the moving frame [e1,e2,e3] of the curve r. The amount of frames to
    be plotted can be controlled with "interval".

    Args:
        r: Vector of 3d points (x,y,z) of curve
        e1: Vector of first component of frame
        e2: Vector of second component of frame
        e3: Vector of third component of frame
        interval: Percentage of frames to be plotted, i.e, 1 plots a frame in
                  every point of r, while 0 does not plot any.
        scale: Float to size components of frame
        ax: Axis where plot will be modified

    Returns:
        ax: Modified plot
    """
    # scale = 0.1
    nn = r.shape[0]
    tend = r + e1 * scale
    nend = r + e2 * scale
    bend = r + e3 * scale

    if ax is None:
        ax = plt.figure().add_subplot(111, projection="3d")

    if interval == 1:
        rng = range(nn)
    else:
        rng = range(0, nn, int(nn * (1 - interval)) if nn > 1 else 1)

    if planar:
        for i in rng:  # if nn >1 else 1):
            ax.plot([r[i, 0], tend[i, 0]], [r[i, 1], tend[i, 1]], "r")
            ax.plot([r[i, 0], nend[i, 0]], [r[i, 1], nend[i, 1]], "g")

            # ax.plot([r[i, 0], tend[i, 0]], [r[i, 2], tend[i, 2]], "r")  # , linewidth=2)
            # ax.plot([r[i, 0], bend[i, 0]], [r[i, 2], bend[i, 2]], "g")  # , linewidth=2)
        ax.set_aspect("equal")

    else:
        if ax_equal:
            ax = axis_equal(r[:, 0], r[:, 1], r[:, 2], ax=ax)

        for i in rng:
            ax.plot(
                [r[i, 0], tend[i, 0]], [r[i, 1], tend[i, 1]], [r[i, 2], tend[i, 2]], "r"
            )
            ax.plot(
                [r[i, 0], nend[i, 0]], [r[i, 1], nend[i, 1]], [r[i, 2], nend[i, 2]], "g"
            )
            ax.plot(
                [r[i, 0], bend[i, 0]], [r[i, 1], bend[i, 1]], [r[i, 2], bend[i, 2]], "b"
            )

    return ax


def plotkitti_2d(frame, pcl_sample, car_p_sample, car_R_sample, car_traj, time):
    frame = frame + 2
    plt.clf()
    fig = plt.gcf()
    ax = fig.gca()
    ax.scatter(
        pcl_sample[:, 0],
        pcl_sample[:, 1],
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

    ax.plot(car_traj[:frame, 0], car_traj[:frame, 1], "-m", alpha=1)
    ax.plot(car_traj[frame:, 0], car_traj[frame:, 1], "-m", alpha=0.2)

    ax.set_xlim(-60, 60)
    ax.set_ylim(-100, 80)
    ax.set_aspect("equal")
    ax.set_axis_off()

    plt.title("Frame: " + str(frame) + ", Time: " + str(time) + "s")


def plotkitti_3d(frame, pcl_sample, car_p_sample, car_R_sample, car_traj, time):
    frame = frame + 2

    plt.clf()
    fig = plt.gcf()
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=33, azim=143, roll=0)
    ax.scatter(
        pcl_sample[:, 0],
        pcl_sample[:, 1],
        pcl_sample[:, 2],
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

    ax.plot(
        car_traj[:frame, 0], car_traj[:frame, 1], car_traj[:frame, 2], "-m", alpha=1
    )
    ax.plot(
        car_traj[frame:, 0], car_traj[frame:, 1], car_traj[frame:, 2], "-m", alpha=0.2
    )

    # indx_in = np.argwhere((pcl_sample[:, 0] > -60) & (pcl_sample[:, 0] < 60))[:, 0]
    # indy_in = np.argwhere((pcl_sample[:, 1] > -100) & (pcl_sample[:, 1] < 80))[:, 0]
    # indz_in = np.argwhere((pcl_sample[:, 2] > -2) & (pcl_sample[:, 2] < 2))[:, 0]
    ax = axis_equal(pcl_sample[:, 0], pcl_sample[:, 1], pcl_sample[:, 2], ax)
    ax.set_axis_off()

    plt.title("Frame: " + str(frame) + ", Time: " + str(time) + "s")


def save_animation(animation, file_name):
    file_path = get_afterkitti_path() + "/docs/" + file_name + ".gif"
    print("Saving animation ", file_name, "...")
    animation.save(
        file_path,
        writer="ffmpeg",
    )
    print("Done. Animation saved in ", file_path)
