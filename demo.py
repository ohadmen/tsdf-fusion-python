"""Fuse 1000 RGB-D images from the 7-scenes dataset into a TSDF voxel volume with 2cm resolution.
"""

import time
import numpy as np
from pyzview import Pyzview
import pyrealsense2 as rs
import fusion
from cam_pose_estimator import CamPoseEstimator
import matplotlib.pyplot as plt
if __name__ == "__main__":
    vol_bnds = np.array([[-3, 3], [-2, 2], [-3, 3]])
    res = (480,640)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, res[1], res[0], rs.format.z16, 30)
    config.enable_stream(rs.stream.color, res[1], res[0], rs.format.bgr8, 30)
    cfg = profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    align_to = rs.stream.color
    align = rs.align(align_to)

    profile = cfg.get_stream(rs.stream.depth)  # Fetch stream profile for depth stream
    intr = profile.as_video_stream_profile().get_intrinsics()  # Downcast to video_stream_profile and fetch intrinsics
    k = np.array([[intr.fx,0,intr.ppx],[0,intr.fy,intr.ppy],[0,0,1]])

    pose = CamPoseEstimator(k, res)

    # ======================================================================================================== #
    # Integrate
    # ======================================================================================================== #
    # Initialize voxel volume
    print("Initializing voxel volume...")
    tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.02)

    # Loop through RGB-D images and fuse them together
    t0_elapse = time.time()
    i=0
    while True:

        if Pyzview().get_last_keystroke() == Pyzview().KEY_ESC:
            break

        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_im = aligned_frames.get_depth_frame()
        color_im = aligned_frames.get_color_frame()
        if not depth_im or not color_im:
            continue

        depth_im = np.asanyarray(depth_im.get_data()) * depth_scale
        color_im = np.asanyarray(color_im.get_data())

        if i == 0:
            verts = []
            norms = []
        else:
            verts, faces, norms, colors = tsdf_vol.get_mesh()
            Pyzview().remove_shape("reconstruct")
            Pyzview().add_trimesh("reconstruct", verts, faces.copy(), colors.copy())

        cam_pose = pose.estimate(depth_im, verts, norms)
        # cam_pose = np.loadtxt("data/frame-%06d.pose.txt"%(i))

        # Integrate observation into voxel volume (assume color aligned with depth)
        tsdf_vol.integrate(color_im, depth_im, k, cam_pose, obs_weight=1.)
        i+=1
