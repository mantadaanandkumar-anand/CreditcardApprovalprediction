FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port (7860 for Hugging Face, Render uses PORT environment variable)
EXPOSE 7860

# Run the Flask app with Gunicorn, binding to the port specified by the PORT environment variable (default 7860)
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-7860} app:app"]
