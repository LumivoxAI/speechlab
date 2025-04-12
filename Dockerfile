# syntax=docker/dockerfile:1

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN apt-get update && \
    apt-get install -y --no-install-recommends git build-essential && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

RUN uv pip install .[docker] --system --no-cache && \
    rm -r ./pyproject.toml ./uv.lock ./README.md ./src/

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
COPY src/server_main.py ./

RUN chmod +x /usr/local/bin/entrypoint.sh

VOLUME [ "/models", "/reference" ]
EXPOSE 5501
EXPOSE 5502
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["--docker", "True", "--model_dir", "/model", "--reference_dir", "/reference"]
