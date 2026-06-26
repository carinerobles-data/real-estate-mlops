# =========================
# IMAGE DE BASE
# =========================
FROM python:3.11-slim

# =========================
# WORKDIR
# =========================
WORKDIR /app

# =========================
# DEPENDANCES
# =========================
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# =========================
# CODE APPLICATION
# =========================
COPY src/ src/

# =========================
# PORT API
# =========================
EXPOSE 8000

# =========================
# LANCEMENT
# =========================
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]