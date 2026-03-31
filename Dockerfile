# Use official Python lightweight image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Working directory inside container
WORKDIR /app

# Install dependencies first (for layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything else, ignoring what is in .dockerignore or .gitignore
COPY . .

# Expose FastAPI default port
EXPOSE 8000

# Run Uvicorn server pointing to api.py
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

