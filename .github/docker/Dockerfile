#syntax=docker/dockerfile:1.3
ARG FROM_IMAGE=ubuntu:25.04
FROM ${FROM_IMAGE}

ARG AIT_EXTRAS=dev
ARG ONNXRUNTIME_VERSION=1.16.2
ARG FRAMEWORK_VERSION=25.10

ENV UV_SYSTEM_PYTHON=true
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV UV_CONSTRAINT=""

ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PIP_CONSTRAINT=""

ENV AIT_DIR="/opt/ai-tune"
ENV AIT_DIST_DIR=${AIT_DIR}/dist
ENV AIT_TEST_DIR=${AIT_DIR}/tests
ENV TMP_DIR="/tmp/ai-tune"

COPY dist/*aitune*.whl ${AIT_DIST_DIR}/
COPY dist/*aitune*.tar.gz ${AIT_DIST_DIR}/

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN --mount=type=cache,target=/root/.cache/uv \
    export WHEEL_PATH=$(ls ${AIT_DIST_DIR}/*aitune*.whl) && \
    pip install ${WHEEL_PATH} ${AIT_EXTRAS} && \
    mkdir -p ${TMP_DIR} && \
    tar xvf ${AIT_DIST_DIR}/*.tar.gz --strip-components 1 -C ${TMP_DIR} && \
    mv ${TMP_DIR}/examples ${AIT_DIR} && \
    mv ${TMP_DIR}/tests ${AIT_DIR} && \
    rm -rf ${TMP_DIR}

ARG ADDITIONAL_PACKAGES
ENV ADDITIONAL_PACKAGES=${ADDITIONAL_PACKAGES}
RUN if [ -n "$ADDITIONAL_PACKAGES" ]; then pip install $ADDITIONAL_PACKAGES; fi

COPY tests ${AIT_TEST_DIR}

WORKDIR ${AIT_DIR}
