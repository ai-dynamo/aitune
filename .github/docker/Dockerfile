#syntax=docker/dockerfile:1.3
ARG FROM_IMAGE=nvcr.io/nvidia/pytorch:26.01-py3
FROM ${FROM_IMAGE}

ARG USER_ID=1001
ARG GROUP_ID=1001
ENV _USER_ID=${USER_ID}
ENV _GROUP_ID=${GROUP_ID}

ARG FRAMEWORK_VERSION=26.01

ARG ONNXRUNTIME_VERSION=1.24.4
ENV ONNXRUNTIME_VERSION=${ONNXRUNTIME_VERSION}

ARG ADDITIONAL_PACKAGES
ENV ADDITIONAL_PACKAGES=${ADDITIONAL_PACKAGES}

ENV AIT_DIR="/opt/ai-tune"
ENV AIT_DIST_DIR=${AIT_DIR}/dist
ENV AIT_TEST_DIR=${AIT_DIR}/tests
ENV TMP_DIR="/tmp/ai-tune"

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=dist,target=/aitune_dist \
    export WHEEL_PATH=$(ls /aitune_dist/*aitune*.whl) && \
    mkdir -p ${TMP_DIR} && \
    tar xf /aitune_dist/*.tar.gz --strip-components 1 -C ${TMP_DIR} && \
    mv ${TMP_DIR}/examples ${AIT_DIR} && \
    mv ${TMP_DIR}/tests ${AIT_DIR} && \
    (cd ${TMP_DIR} && pip install . --group test --break-system-packages) && \
    pip install --no-deps ${WHEEL_PATH} --break-system-packages && \
    rm -rf ${TMP_DIR} && \
    pip install pydantic tomli --break-system-packages && \
    if [ -n "$ADDITIONAL_PACKAGES" ]; then pip install --break-system-packages $ADDITIONAL_PACKAGES; fi && \
    pip install onnxruntime-gpu==$ONNXRUNTIME_VERSION --no-deps --force-reinstall --break-system-packages --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/ && \
    groupadd -g $_GROUP_ID runner && useradd -m -u $_USER_ID -g $_GROUP_ID runner && \
    chown -R runner:runner ${AIT_DIR} && \
    chmod -R ug+rw ${AIT_DIR}

USER runner

WORKDIR ${AIT_DIR}
