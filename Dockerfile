# Stage 1: Build the environment with dependencies
# Use an official Python runtime as a parent image
FROM python:3.10-slim-bullseye as builder

# Set the working directory in the container
WORKDIR /app

# Prevent python from writing pyc files to disc
ENV PYTHONDONTWRITEBYTECODE 1
# Ensure python output is sent straight to the terminal without buffering
ENV PYTHONUNBUFFERED 1

# Install system dependencies required by some Python packages (e.g., OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
# Copy only the requirements file to leverage Docker layer caching
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt


# Stage 2: Create the final production image
# Use a slim Python image for a smaller final image size
FROM python:3.10-slim-bullseye

# Set the working directory
WORKDIR /app

# Create a non-root user and group
RUN addgroup --system app && adduser --system --ingroup app appuser

# Install system dependencies from the builder stage (if any were needed beyond base)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the pre-built wheels from the builder stage and install them
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/* && rm -rf /wheels

# Copy the application source code
COPY ./src ./src

# Copy trained models and other necessary assets
# NOTE: This assumes your trained models are in a top-level 'models' directory.
# Adjust the path if your models are stored elsewhere.
COPY ./models ./models

# Set ownership of the app directory to the non-root user
RUN chown -R appuser:app /app

# Switch to the non-root user
USER appuser

# Expose the port the app runs on
EXPOSE 8000

# Define the command to run the application
# The API entrypoint is src/deepfake_detector/api/main.py, with the FastAPI instance 'app'
CMD ["uvicorn", "src.deepfake_detector.api.main:app", "--host", "0.0.0.0", "--port", "8000"]