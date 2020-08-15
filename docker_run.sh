#!/bin/bash
# Description of the Docker command-line options shown below:
# --mount type=bind,source=$PWD,target=/data: Mounts the current directory at /data in container.
# --name: Gives a name to this container for easy reference.
# --workdir: Sets the working directory inside the container.

docker run --rm -it --init \
  --mount type=bind,src=$PWD/images,dst=/data \
  --name fisheye \
  --workdir /opt/fisheye-distortion \
  fisheye:1.0 /bin/bash
