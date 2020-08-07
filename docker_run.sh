# Description of the Docker command-line options shown below:
#--init: Specifying an init process ensures the usual responsibilities of an init system, such as reaping zombie processes, are performed inside the created container
#--mount type=bind,source=$PWD,target=/data: Mounts the current direct into /data in container.
#--name cloud: Gives a name to this container for easy reference

docker run --rm -it --init \
  --mount type=bind,src=$PWD/images,dst=/data \
  --name fisheye \
  fisheye:1.0 /bin/bash
