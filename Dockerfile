# Runtime / base image
FROM python:3.13-slim
WORKDIR /app

# Python + dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Serving code + training code
COPY . .

# Model: train it inside the image, so the image is complete and versioned
RUN python train.py

# Start the API (Render sets the PORT variable; 8000 is the default)
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
