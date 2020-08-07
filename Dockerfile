FROM python:3.8

COPY requirements.txt /opt/fisheye-distortion/requirements.txt
RUN pip install --no-cache-dir -r /opt/fisheye-distortion/requirements.txt
COPY . /opt/fisheye-distortion
