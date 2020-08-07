FROM python:3.8

#RUN apt-get update && apt-get install -y \
#	libglib2.0-0 \
# 	libopenexr-dev \
#	libsm6 \
#	libxext-dev \
#	libxrender1 \
#	zlib1g-dev \
# && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /opt/fisheye-distortion/requirements.txt
RUN pip install --no-cache-dir -r /opt/fisheye-distortion/requirements.txt
COPY . /opt/fisheye-distortion
CMD tail -f /dev/null
