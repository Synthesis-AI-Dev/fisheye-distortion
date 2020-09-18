# fisheye-distortion

![Method](images/fisheye-distort-fig.png)

This script applies a fisheye distortion to rendered images. The output images can be cropped to a
rectangle and resized. The fisheye distortion is the same as detailed in OpenCV 4.4 [docs](https://docs.opencv.org/4.4.0/db/d58/group__calib3d__fisheye.html).
As mentioned in the docs, the distortion is characterized by 4 parameters, `[k1, k2, k3, k4]`, which are 
passed via the config file or command line arguments.

## Usage
The arguments for the script are passed via the config file. All the parameters in the
 config file can be overriden from the command line:
 
```bash
python apply_fisheye_distortion.py dir_input=samples input.info=".info.json"
```

Alternately, you can use the provided dockerfile. 

```bash
# Build the docker image
bash docker_build.sh

# Run the docker image
bash docker_run.sh

# Inside container
$ python apply_fisheye_distortion.py dir_input=/data
```

This will process all the images found in the container's `/data` directory. Modify the
`docker_run.sh` script to change which host directory is mounted to the container's `/data`.

### Config file
You can edit the config file, `config.yaml`, to customize the default parameters:

```yaml
dir_input: samples/
dir_output: samples/
input:
  # Distortion will be applied to these files (except info).
  # Set their name to null to disable distorting those images
  rgb: .rgb.png
  segments: .segments.png  # Mask of all objects in scene. 8 or 16 bit PNG.
  info: .json  # Contains camera intrinsics

workers: 0  # How many workers to use to convert files. 0 for all cores.

crop_and_resize_output: True  # Disable to get original distorted image

distortion_parameters:
  # Ref: https://docs.opencv.org/4.4.0/db/d58/group__calib3d__fisheye.html
  # These are the distortion parameters of the fisheye camera model as defined in the fisheye module of OpenCV 4.4.0
  k1: 0.17149
  k2: -0.27191
  k3: 0.25787
  k4: -0.08054

hydra:
    output_subdir: null  # Disable saving of config files. We'll do that ourselves.
    run:
        dir: .  # Set working dir to current directory
```

## Install
This script requires Python 3.7 or greater. The dependencies can be installed via pip:

```bash
pip install requirements.txt
```

## Detailed Description
From OpenCV [docs](https://docs.opencv.org/4.4.0/db/d58/group__calib3d__fisheye.html),
here is the description of the fisheye camera model used to create the distortion. The
distortion co-efficients `[k1, k2, k3, k4]` are detailed below:
![distortion-description](images/fisheye-opencv-description.png)
