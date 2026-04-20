FROM python:3.11-slim

RUN useradd -m -u 1000 user
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY webapp_requirements.txt .
RUN pip install --no-cache-dir -r webapp_requirements.txt

COPY --chown=user:user models/ ./models/
COPY --chown=user:user webapp/ ./webapp/

ENV PORT=7860
ENV DB_PATH=/data/boxing.db

USER user
EXPOSE 7860

CMD ["uvicorn", "webapp.main:app", "--host", "0.0.0.0", "--port", "7860"]
