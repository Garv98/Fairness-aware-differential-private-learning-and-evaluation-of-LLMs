# FairDP demo backend: FastAPI + PyTorch (CPU).
# Runs on Hugging Face Spaces (Docker SDK, port 7860), Render, Railway, or locally:
#   docker build -t fairdp-api . && docker run -p 7860:7860 -e GROQ_API_KEY=... fairdp-api
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PYTHONWARNINGS=ignore
RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements-server.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install -r requirements-server.txt

COPY --chown=user . .
USER user

# Train + evaluate the served models at build time (downloads UCI Adult from OpenML),
# so the container starts with all artifacts ready.
RUN cd server && python train_models.py

ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app:app --app-dir server --host 0.0.0.0 --port ${PORT}"]
