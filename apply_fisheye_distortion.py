import concurrent.futures
import enum
import itertools
import json
import logging
from pathlib import Path

import cv2
import hydra
import numpy as np
import scipy.interpolate
import tifffile
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

CONFIG_FILE = 'config.yaml'


class DistortMode(enum.Enum):
    LINEAR = 'linear'
    NEAREST = 'nearest'


def distort_image(img: np.ndarray, cam_intr: np.ndarray, dist_coeff: np.ndarray,
                  mode: DistortMode = DistortMode.LINEAR, crop_output: bool = True) -> np.ndarray:
    """Apply fisheye distortion to an image

    Args:
        img (numpy.ndarray): BGR image. Shape: (H, W, 3)
        cam_intr (numpy.ndarray): The camera intrinsics matrix, in pixels: [[fx, 0, cx], [0, fx, cy], [0, 0, 1]]
                            Shape: (3, 3)
        dist_coeff (numpy.ndarray): The fisheye distortion coefficients, for OpenCV fisheye module.
                            Shape: (1, 4)
        mode (DistortMode): For distortion, whether to use nearest neighbour or linear interpolation.
                            RGB images = linear, Mask/Surface Normals/Depth = nearest
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
    interpolators = [scipy.interpolate.RegularGridInterpolator((ys, xs), img[:, :, channel], method=mode.value,
                                                               bounds_error=False, fill_value=0)
                     for channel in range(chan)]
    img_dist = np.dstack([interpolator(undistorted_px) for interpolator in interpolators])

    if imdtype == np.uint8:
        # RGB Image
        img_dist = img_dist.round().clip(0, 255).astype(np.uint8)
    elif imdtype == np.uint16:
        # Mask
        img_dist = img_dist.round().clip(0, 65535).astype(np.uint16)
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


def _process_file(f_json: Path, f_img: Path, dir_output: Path, dist_coeff: np.ndarray, mode: DistortMode,
                  crop_resize_output: bool):
    """Apply fisheye effect to file and save output
    Args:
        f_json (Path): Json file containing camera intrinsics
        f_img (Path): Image to distort
        dir_output (Path): Which dir to store outputs in
        dist_coeff (numpy.ndarray): The distortion coefficients. Shape: (1, 4).
        mode (DistortMode): Which type of interpolation to use for distortion.
                              - Linear -> RGB images
                              - Nearest -> Masks, Surface Normals and Depth images
        crop_resize_output (bool): Whether the output should be cropped to rectange and resized to original dimensions
    """
    # Load Camera intrinsics and RGB image
    with f_json.open() as json_file:
        metadata = json.load(json_file)
        metadata = OmegaConf.create(metadata)
    K = np.array(metadata.camera.intrinsics, dtype=np.float32)
    img = cv2.imread(str(f_img), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    # Apply distortion
    dist_img = distort_image(img, K, dist_coeff, mode, crop_output=crop_resize_output)
    if crop_resize_output:
        h, w = img.shape[:2]
        dist_img = cv2.resize(dist_img, (w, h), cv2.INTER_CUBIC)

    # Save Result
    out_filename = dir_output / f"{f_img.stem}.dist{f_img.suffix}"
    if f_img.suffix == '.tif' or f_img.suffix == '.tiff':
        tifffile.imsave(out_filename, dist_img, compress=1)
    else:
        retval = cv2.imwrite(str(out_filename), dist_img)
        if not retval:
            raise RuntimeError(f'Error in saving file {out_filename}')


@hydra.main(config_path='.', config_name='config')
def main(cfg: DictConfig):
    """This script creates fisheye distortion in images, using the OpenCV 4.4 fisheye camera model
    Look at equations in detailed description at: https://docs.opencv.org/4.4.0/db/d58/group__calib3d__fisheye.html

    After distortion, the image size will be reduced. To counter that, we resize the output back to original dimensions.
    The parameters in config file can be modified from the command line.
    """
    log = logging.getLogger(__name__)

    # Read Distortion Parameters
    dist = cfg.distortion_parameters
    dist_coeff = np.array([dist.k1, dist.k2, dist.k3, dist.k4])
    log.info(f'Loaded distortion coefficients: {dist_coeff}')

    crop_resize_output = cfg.crop_and_resize_output
    log.info(f'Crop and Resize output: {crop_resize_output}')
    if cfg.linear_interpolation is True:
        interpolate_mode = DistortMode.LINEAR
    elif cfg.linear_interpolation is False:
        interpolate_mode = DistortMode.NEAREST
    else:
        raise ValueError(f'Linear interpolation must be True or False. Given: {cfg.linear_interpolation}')
    log.info(f'Interpolation mode for distortion: {interpolate_mode}')

    if int(cfg.workers) > 0:
        max_workers = int(cfg.workers)
    else:
        max_workers = None

    # Read input/output parameters
    dir_input = Path(cfg.dir.input)
    if not dir_input.is_dir():
        raise ValueError(f'Not a directory: {dir_input}')
    log.info(f'Input Dir: {dir_input}')

    if cfg.dir.output is None:
        dir_output = dir_input
    else:
        dir_output = Path(cfg.dir.output)
        if not dir_output.exists():
            dir_output.mkdir(parents=True)
    log.info(f'Output Dir: {dir_output}')

    ext_input = cfg.file_ext.input
    log.info(f'Input File Ext: {ext_input}')

    ext_info = cfg.file_ext.info
    ext_info_type = ext_info.split('.')[-1]
    if ext_info_type != 'json':
        raise ValueError(f'Unsupported filetype: {ext_info_type}. Info files must be of type json')

    info_filenames = sorted(dir_input.glob('*' + ext_info))
    num_json = len(info_filenames)

    input_filenames = sorted(dir_input.glob('*' + ext_input))
    num_images = len(input_filenames)
    if num_images < 1:
        raise ValueError(f'No images found. Searched:\n'
                         f'  dir: "{dir_input}"\n'
                         f'  file extention: "{ext_input}"')
    elif num_images != num_json:
        raise ValueError(f'Unequal number of json files ({num_json}) and images ({num_images}). Searched:\n'
                         f'  dir: "{dir_input}"\n'
                         f'  image file extention: "{ext_input}"')
    log.info(f'Num Input Files: {num_images}')

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(info_filenames)) as pbar:
            for _ in executor.map(_process_file, info_filenames, input_filenames, itertools.repeat(dir_output),
                                  itertools.repeat(dist_coeff), itertools.repeat(interpolate_mode),
                                  itertools.repeat(crop_resize_output)):
                # Catch any error raised in processes
                pbar.update()


if __name__ == "__main__":
    main()
