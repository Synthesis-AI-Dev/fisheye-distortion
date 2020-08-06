"""This script creates fisheye distortion in images, using the OpenCV 4.4 fisheye camera model

Look at equations in detailed description at:
https://docs.opencv.org/4.4.0/db/d58/group__calib3d__fisheye.html
Note: fisheye is a new module that is different from old OpenCV 2.4 distortion equations
"""
import argparse

import cv2
import numpy as np
import scipy.interpolate


parser = argparse.ArgumentParser(description='Apply fisheye effect to all matching images in a directory')
parser.add_argument('-i', '--path_img', required=True, help='Path to undistorted input image', metavar='path/to/img.png')
parser.add_argument('-o', '--path_img_out', required=True, help='Path to save output image', metavar='path/to/img.png')
parser.add_argument('-k', '--path_intr', required=True, help='Path to camera intrinsics .txt', metavar='path/to/intr.txt')
args = parser.parse_args()
path_img = args.path_img
path_img_out = args.path_img_out
path_intr = args.path_intr

k1 = 0.17149
k2 = -0.27191
k3 = 0.25787
k4 = -0.08054
D = np.array([k1, k2, k3, k4], dtype=np.float32)
K = np.loadtxt(path_intr, dtype=np.float32)

img = cv2.imread(path_img)
h, w, _ = img.shape

# Get array of pixel co-ords
xs = np.arange(w)
ys = np.arange(h)
xv, yv = np.meshgrid(xs, ys)
img_pts = np.stack((xv, yv), axis=0)  # shape (2, H, W)
img_pts = img_pts.reshape((2, -1))  # shape: (2, N)

img_pts = np.transpose(img_pts, (1, 0))  # Shape: (N, 2)
img_pts = np.expand_dims(img_pts, axis=0)  # shape: (1, N, 2)

undistorted_px = cv2.fisheye.undistortPoints(img_pts, K, D)  # shape: (1, N, 2)
undistorted_px = np.squeeze(undistorted_px, axis=0)  # Shape: (N, 2)
undistorted_px = np.transpose(undistorted_px, (1, 0))  # Shape: (2, N)
undistorted_px = undistorted_px.reshape((2, h, w))  # Shape: (2, H, W)
undistorted_px = np.transpose(undistorted_px, (1, 2, 0))  # Shape: (H, W, 2)
undistorted_px = np.flip(undistorted_px, axis=2)  # flip x, y coordinates of the points as cv2 is height first

# Map RGB values from input img using distorted pixel co-ordinates
interpolators = [scipy.interpolate.RegularGridInterpolator((ys, xs), img[:, :, chanel], bounds_error=False, fill_value=0)
                 for chanel in range(3)]
img_dist = np.dstack([interpolator(undistorted_px) for interpolator in interpolators])
img_dist = img_dist.clip(0, 255).astype(np.float32)

cv2.imwrite(path_img_out, img_dist)
print(f'exported image: {path_img_out}')