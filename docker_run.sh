# Here's a description of the Docker command-line options shown below:
#--init: Specifying an init process ensures the usual responsibilities of an init system, such as reaping zombie processes, are performed inside the created container
#--gpus=all: Required if using CUDA, optional otherwise. Passes the graphics cards from the host to the container. You can also more precisely control which graphics cards are exposed using this option (see documentation at https://github.com/NVIDIA/nvidia-docker).
#--ipc=host: Required if using multiprocessing, as explained at https://github.com/pytorch/pytorch#docker-image.
#--user="$(id -u):$(id -g)": Sets the user inside the container to match your user and group ID. Optional, but is useful for writing files with correct ownership.
#--mount type=bind,source=$PWD,target=/data: Mounts the current direct into /data in container.
#--name cloud: Gives a name to this container for easy reference

# Due to the user ID change installing packages with apt does not work.
# If you need additional packages installed or updated with apt, open a second terminal and run:
# docker exec -u root <containername> apt update
# and then this to update the packages as root in the running container:
# docker exec -u root <containername> apt upgrade -y

docker run --rm -it --init \
  --ipc=host \
  --mount type=bind,src=$PWD/images,dst=/data \
  --name fisheye \
  fisheye:1.0 /bin/bash
