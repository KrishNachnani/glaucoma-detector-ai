FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TZ=UTC

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .


# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy models directory first (specifically for model files)
COPY models/ /app/models/

# Copy the rest of the project
COPY . .

# Create necessary directories for uploads and logs
RUN mkdir -p logs uploaded/glaucoma uploaded/noglaucoma

# Ensure model files have correct permissions
RUN chmod -R 755 /app/models/

# Make sure the application logs are saved
VOLUME ["/app/logs"]
VOLUME ["/app/uploaded"]

# Expose the port that the app will run on
EXPOSE 8001

# Command to run the application
CMD ["python", "app.py"]