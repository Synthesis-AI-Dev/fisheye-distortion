"""This script creates fisheye distortion in images, using the OpenCV 4.4 fisheye camera model

Look at equations in detailed description at:
https://docs.opencv.org/4.4.0/db/d58/group__calib3d__fisheye.html
Note: fisheye is a new module that is different from old OpenCV 2.4 distortion equations

The dir of images to convert must contain the camera intrinsics, in pixels, in a text file.
"""
import argparse
import yaml
from pathlib import Path

import cv2
import numpy as np
import scipy.interpolate

CAMERA_INTR_FILE = 'camera_intrinsics.txt'


def distort_image(img: np.ndarray, cam_intr: np.ndarray, dist_coeff: np.ndarray) -> np.ndarray:
    """Apply fisheye distortion to an image

    Args:
        img (numpy.ndarray): BGR image. Shape: (H, W, 3)
        cam_intr (numpy.ndarray): The camera intrinsics matrix, in pixels: [[fx, 0, cx], [0, fx, cy], [0, 0, 1]]
                            Shape: (3, 3)
        dist_coeff (numpy.ndarray): The fisheye distortion coefficients, for OpenCV fisheye module.
                            Shape: (1, 4)

    Returns:
        numpy.ndarray: The distorted image, same resolution as input image. Unmapped pixels will be black in color.
    """
    assert cam_intr.shape == (3, 3)
    assert dist_coeff.shape == (4,)

    h, w, _ = img.shape

    # Get array of pixel co-ords
    xs = np.arange(w)
    ys = np.arange(h)
    xv, yv = np.meshgrid(xs, ys)
    img_pts = np.stack((xv, yv), axis=2)  # shape (H, W, 2)
    img_pts = img_pts.reshape((-1, 1, 2)).astype(np.float32)  # shape: (N, 1, 2)

    # Get the mapping from distorted pixels to undistorted pixels
    undistorted_px = cv2.fisheye.undistortPoints(img_pts, cam_intr, dist_coeff)  # shape: (N, 1, 2)
    undistorted_px = cv2.convertPointsToHomogeneous(undistorted_px)  # Shape: (N, 1, 3)
    undistorted_px = np.tensordot(undistorted_px, cam_intr, axes=(2, 1))  # To camera coordinates, Shape: (N, 1, 3)
    undistorted_px = cv2.convertPointsFromHomogeneous(undistorted_px)  # Shape: (N, 1, 2)
    undistorted_px = undistorted_px.reshape((h, w, 2))  # Shape: (H, W, 2)
    undistorted_px = np.flip(undistorted_px, axis=2)  # flip x, y coordinates of the points as cv2 is height first

    # Map RGB values from input img using distorted pixel co-ordinates
    interpolators = [scipy.interpolate.RegularGridInterpolator((ys, xs), img[:, :, chanel], bounds_error=False, fill_value=0)
                     for chanel in range(3)]
    img_dist = np.dstack([interpolator(undistorted_px) for interpolator in interpolators])
    img_dist = img_dist.clip(0, 255).astype(np.uint8)

    # Crop rectangle from resulting distorted image
    # Get mapping from undistorted to distorted
    distorted_px = cv2.convertPointsToHomogeneous(img_pts)  # Shape: (N, 1, 3)
    cam_intr_inv = np.linalg.inv(cam_intr)
    distorted_px = np.tensordot(distorted_px, cam_intr_inv, axes=(2, 1))  # To camera coordinates, Shape: (N, 1, 3)
    distorted_px = cv2.convertPointsFromHomogeneous(distorted_px)  # Shape: (N, 1, 2)
    distorted_px = cv2.fisheye.distortPoints(distorted_px, cam_intr, dist_coeff)  # shape: (N, 1, 2)
    distorted_px = distorted_px.reshape((h, w, 2))
    # Get the corners. Round values up/down accordingly to avoid invalid pixel selection.
    top_left = np.ceil(distorted_px[0, 0, :]).astype(np.int)
    bottom_right = np.floor(distorted_px[(h - 1), (w - 1), :]).astype(np.int)
    img_dist = img_dist[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], :]

    return img_dist


def main(args):
    dir_images = Path(args.dir_images)
    dir_output = Path(args.dir_output)
    config_file = Path(args.config_file)
    ext_images = args.ext_images

    if not dir_images.exists() or not dir_images.is_dir():
        raise ValueError(f'Not a directory: {dir_images}')
    if not config_file.is_file():
        raise ValueError(f'Not a file: {config_file}')
    if not dir_output.exists():
        dir_output.mkdir(parents=True)

    image_filenames = sorted(list(dir_images.glob('*' + ext_images)))
    num_images = len(image_filenames)
    if num_images < 1:
        raise ValueError(f'No images found in dir {dir_images}, matching extention {ext_images}')
    else:
        print(f'Found {num_images} images. Applying distortion')

    camera_intr_file = dir_images / CAMERA_INTR_FILE
    K = np.loadtxt(str(camera_intr_file))
    print(f'Loaded camera intrinsics: \n{K}')

    with open(config_file) as fd:
        dist = yaml.load(fd, Loader=yaml.Loader)
    D = np.array([dist['k1'], dist['k2'], dist['k3'], dist['k4']])
    print(f'Loaded distortion coefficients: {D}')

    for f_img in image_filenames:
        img = cv2.imread(str(f_img))
        dist_img = distort_image(img, K, D)
        out_filename = dir_output / f"{f_img.stem}.dist{f_img.suffix}"
        retval = cv2.imwrite(str(out_filename), dist_img)
        if retval:
            print(f'exported image: {out_filename}')
        else:
            raise RuntimeError(f'Error in saving file {out_filename}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply fisheye effect to all matching images in a directory')
    parser.add_argument('-i', '--dir_images', type=Path, required=True,
                        help='Directory containing the images to distort',metavar='path/to/dir')
    parser.add_argument('-e', '--ext_images', default='.rgb.png', help='Filename extention of images to convert')
    parser.add_argument('-o', '--dir_output', type=Path, required=True,
                        help='Path to save output images', metavar='path/to/dir')
    parser.add_argument('-c', '--config_file', default='distortion_parameters.yaml',
                        help='Path to config file with distortion params', metavar='path/to/config.yaml')
    parser.add_argument("-w", "--workers", type=int, default=0,
                        help="Number of processes to use. Defaults to the number of processors on the machine.")
    args = parser.parse_args()
    main(args)
