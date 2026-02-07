FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (7860 for HuggingFace Spaces, 8000 for others)
EXPOSE 7860
EXPOSE 8000

ENV HOST=0.0.0.0

# Run the application - use PORT env var if set, default to 7860 for HF Spaces
CMD ["sh", "-c", "python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]
