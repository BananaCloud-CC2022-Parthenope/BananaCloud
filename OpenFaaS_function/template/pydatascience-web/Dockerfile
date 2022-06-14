ARG watchdog_version=0.8.4
ARG python_version=3.9
FROM --platform=${TARGETPLATFORM:-linux/amd64} ghcr.io/openfaas/of-watchdog:${watchdog_version} as watchdog
FROM --platform=${TARGETPLATFORM:-linux/amd64} python:${python_version}-slim

COPY --from=watchdog /fwatchdog /usr/bin/fwatchdog
RUN chmod +x /usr/bin/fwatchdog

# Allows you to add additional packages via build-arg
ARG ADDITIONAL_PACKAGE
ARG CHANNEL=pytorch

RUN addgroup app && adduser app --system --ingroup app \
    && mkdir -p /opt/conda && chown -R app /opt/conda \
    && chown -R app /root && chmod -R go+rX /root

ENV HOME /home/app
ENV PATH=$HOME/conda/bin:$PATH

RUN apt-get update \
    && apt-get -y install curl bzip2 ${ADDITIONAL_PACKAGE}
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN curl -sSL "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" -o /tmp/miniforge.sh \
    && chown app /tmp/miniforge.sh \
    && apt-get -qq -y remove curl \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log

# Add non root user
WORKDIR /home/app/
USER app

RUN bash /tmp/miniforge.sh -bfp $HOME/conda \
    && conda install -y python=3 \
    && conda update conda \
    && conda clean --all --yes \
    && rm -rf /tmp/miniforge.sh

COPY requirements.txt   .
RUN conda install --file requirements.txt -c ${CHANNEL} \
    && pip install tox-current-env
RUN conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cpuonly -c pytorch
RUN pip install opencv-python
RUN pip install prometheus-client
COPY index.py           .

RUN mkdir -p function
RUN touch ./function/__init__.py

WORKDIR /home/app/function/
COPY function/requirements*.txt	./
RUN conda install --file requirements.txt --file requirements_test.txt -c ${CHANNEL}


COPY function/ .

ARG TEST_COMMAND="tox --current-env"
ARG TEST_ENABLED=true
RUN if [ "x$TEST_ENABLED" = "xfalse" ]; then \
    echo "skipping tests";\
    else \
    eval "$TEST_COMMAND"; \
    fi

WORKDIR /home/app/
COPY function/ ./function

# Set environment variables
ENV fprocess="python index.py"
ENV cgi_headers="true"
ENV mode="http"
ENV upstream_url="http://127.0.0.1:5000"
ENV function_root="/home/app/function"

HEALTHCHECK --interval=5s CMD [ -e /tmp/.lock ] || exit 1

CMD ["fwatchdog"]
