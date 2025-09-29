FROM python:3.11-slim AS base
COPY . .

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl 

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"
WORKDIR /HM4

CMD ["uv","run", "pytest"]
