FROM occlum/occlum:0.29.2-ubuntu20.04 as base


FROM python:3.8 as python
# "grpcio>=1.28.1,<=1.43.0" couldn't be found in default index url, so install it in advance
RUN pip install -i http://mirrors.cloud.aliyuncs.com/pypi/simple/ --trusted-host mirrors.cloud.aliyuncs.com "grpcio>=1.28.1,<=1.43.0" matplotlib numpy pillow && pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ray[tune]==1.13.0 && python -c "import ray; obj = ray.put(1); print(ray.get(obj));" && pip uninstall -y ray
COPY ray-1.13.0-cp38-cp38-linux_x86_64.whl .
RUN pip install ray-1.13.0-cp38-cp38-linux_x86_64.whl
RUN rm ray-1.13.0-cp38-cp38-linux_x86_64.whl

FROM base
COPY --from=python / /root/image
RUN occlum new occlum_instance
WORKDIR /root/occlum_instance
COPY python-ray.yaml /tmp/python-ray.yaml
COPY pytorch_ray.py /tmp/pytorch_ray.py
RUN rm -rf image && copy_bom -f /tmp/python-ray.yaml --root image --include-dir /opt/occlum/etc/template
COPY Occlum.custom.json /tmp/Occlum.custom.json
RUN jq -s '.[0] * .[1]' Occlum.json /tmp/Occlum.custom.json > Occlum.new.json && mv Occlum.new.json Occlum.json
RUN occlum build

