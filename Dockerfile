# Multi-arch image (linux/amd64 and linux/arm64):
#   docker buildx build --platform linux/amd64,linux/arm64 -t valqyrie .
#   docker run --rm valqyrie python -m VQE --molecule LiH --ansatz uccsd
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY qcore ./qcore
COPY VQE ./VQE
COPY GQE ./GQE
COPY stress ./stress
COPY tests ./tests
COPY examples ./examples
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install ".[test]"
CMD ["python", "-m", "pytest", "-q", "-m", "not slow"]
