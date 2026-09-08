#syntax=docker/dockerfile:1.3
ARG FROM_IMAGE=nvcr.io/nvidia/cuda:12.9.2-devel-ubuntu26.04
FROM ${FROM_IMAGE}

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN groupadd -g 1001 runner && useradd -m -u 1001 -g 1001 runner

USER runner
