import concurrent.futures
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf
import scipy.interpolate
import tifffile

CONFIG_FILE = 'config.yaml'


def distort_image(img: np.ndarray, cam_intr: np.ndarray, dist_coeff: np.ndarray, crop_output: bool = True) -> np.ndarray:
    """Apply fisheye distortion to an image

    Args:
        img (numpy.ndarray): BGR image. Shape: (H, W, 3)
        cam_intr (numpy.ndarray): The camera intrinsics matrix, in pixels: [[fx, 0, cx], [0, fx, cy], [0, 0, 1]]
                            Shape: (3, 3)
        dist_coeff (numpy.ndarray): The fisheye distortion coefficients, for OpenCV fisheye module.
                            Shape: (1, 4)
        crop_output (bool): Whether to crop the output distorted image into a rectangle. The 4 corners of the input
                            image will be mapped to 4 corners of the distorted image for cropping.

    Returns:
        numpy.ndarray: The distorted image, same resolution as input image. Unmapped pixels will be black in color.
    """
    assert cam_intr.shape == (3, 3)
    assert dist_coeff.shape == (4,)

    imshape = img.shape
    if len(imshape) == 3:
        h, w, chan = imshape
    elif len(imshape) == 2:
        h, w = imshape
        chan = 1
    else:
        raise RuntimeError(f'Image has unsupported shape: {imshape}. Valid shapes: (H, W), (H, W, N)')

    imdtype = img.dtype

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
    if chan == 1:
        img = np.expand_dims(img, 2)
    interpolators = [scipy.interpolate.RegularGridInterpolator((ys, xs), img[:, :, channel], bounds_error=False, fill_value=0)
                     for channel in range(chan)]
    img_dist = np.dstack([interpolator(undistorted_px) for interpolator in interpolators])

    if imdtype == np.uint8:
        # RGB Image
        img_dist = img_dist.clip(0, 255).astype(np.uint8)
    if imdtype == np.uint16:
        # Mask
        img_dist = img_dist.clip(0, 65535).astype(np.uint16)
    elif imdtype == np.float16 or imdtype == np.float32 or imdtype == np.float64:
        img_dist = img_dist.astype(imdtype)
    else:
        raise RuntimeError(f'Unsupported dtype for image: {imdtype}')

    if crop_output:
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

    if chan == 1:
        img_dist = img_dist[:, :, 0]

    return img_dist


def process_file(f_json: Path, f_img: Path, dir_output: Path, dist_coeff: np.ndarray, crop_output: bool, resize_h: int,
                 resize_w: int):
    """Apply fisheye effect to file and save output
    Args:
        f_json (Path): Json file containing camera intrinsics
        f_img (Path): Image to distort
        dir_output (Path): Which dir to store outputs in
        dist_coeff (numpy.ndarray): The distortion coefficients. Shape: (1, 4).
        crop_output (bool): Whether the output should be cropped
        resize_w (int): The width to resize distorted image to. Pass 0 to disable resize.
        resize_h (int): The height to resize distorted image to. Pass 0 to disable resize.
    """
    # Load Camera intrinsics and RGB image
    with f_json.open() as json_file:
        metadata = json.load(json_file)
        metadata = OmegaConf.create(metadata)
    K = np.array(metadata.camera.intrinsics, dtype=np.float32)
    img = cv2.imread(str(f_img), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    # Apply distortion
    dist_img = distort_image(img, K, dist_coeff, crop_output=crop_output)
    if resize_h > 0 and resize_w > 0:
        dist_img = cv2.resize(dist_img, (resize_w, resize_h), cv2.INTER_CUBIC)

    # Save Result
    out_filename = dir_output / f"{f_img.stem}.dist{f_img.suffix}"
    if f_img.suffix == '.tif' or f_img.suffix == '.tiff':
        tifffile.imsave(out_filename, dist_img, compress=1)
    else:
        retval = cv2.imwrite(str(out_filename), dist_img)
        if retval:
            print(f'exported image: {out_filename}')
        else:
            raise RuntimeError(f'Error in saving file {out_filename}')


def main():
    """This script creates fisheye distortion in images, using the OpenCV 4.4 fisheye camera model
    Look at equations in detailed description at: https://docs.opencv.org/4.4.0/db/d58/group__calib3d__fisheye.html

    The dir of images to convert must contain the camera intrinsics, in pixels, in a text file.
    The dimensions of output of distortion is ~58% of input image and may not be of the exact same aspect ratio due
    to rounding of pixel co-ordinate. To counter that, we resize the output according to config file.

    The parameters in config file can be modified from the command line.
    """
    base_conf = OmegaConf.load(CONFIG_FILE)
    cli_conf = OmegaConf.from_cli()
    conf = OmegaConf.merge(base_conf, cli_conf)

    dir_input = Path(conf.dir_input)
    dir_output = Path(conf.dir_output)
    input_rgb_ext = conf.input_rgb_ext
    input_json_ext = conf.input_json_ext
    if not dir_input.is_dir():
        raise ValueError(f'Not a directory: {dir_input}')
    if not dir_output.exists():
        dir_output.mkdir(parents=True)

    image_filenames = sorted(dir_input.glob('*' + input_rgb_ext))
    json_filenames = sorted(dir_input.glob('*' + input_json_ext))
    num_images = len(image_filenames)
    num_json = len(json_filenames)
    if num_images < 1:
        raise ValueError(f'No images found in dir {dir_input} that match file extention {input_rgb_ext}')
    elif num_images != num_json:
        raise ValueError(f'The number of RGB images ({num_images}) and json files ({num_json}) is not equal.')
    else:
        print(f'Found {num_images} images. Applying distortion')

    dist = conf.distortion_parameters
    D = np.array([dist.k1, dist.k2, dist.k3, dist.k4])
    print(f'Loaded distortion coefficients: {D}')

    crop_output = conf.dev.crop_output
    resize_h = int(conf.resize_output.h)
    resize_w = int(conf.resize_output.w)

    if int(conf.workers) > 0:
        max_workers = int(conf.workers)
    else:
        max_workers = None
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for _ in executor.map(process_file, json_filenames, image_filenames, itertools.repeat(dir_output),
                                        itertools.repeat(D), itertools.repeat(crop_output), itertools.repeat(resize_h),
                                        itertools.repeat(resize_w)):
            # Catch any error raised in processes
            pass


if __name__ == "__main__":
    main()
