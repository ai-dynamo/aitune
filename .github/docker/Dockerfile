#syntax=docker/dockerfile:1.3
ARG FROM_IMAGE=nvcr.io/nvidia/pytorch:25.10-py3
FROM ${FROM_IMAGE}

ARG ONNXRUNTIME_VERSION=1.16.2
ARG FRAMEWORK_VERSION=25.10

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
    pip install pydantic tomli && \
    if [ -n "$ADDITIONAL_PACKAGES" ]; then pip install $ADDITIONAL_PACKAGES; fi && \
    groupadd -g 1001 runner && useradd -m -u 1001 -g 1001 runner && \
    chown -R runner:runner ${AIT_DIR} && \
    chmod -R ug+rw ${AIT_DIR}

USER runner

WORKDIR ${AIT_DIR}
