# Use the official Python image from Docker Hub
FROM python:3.9-slim

# Set environment variables to prevent Python from writing .pyc files to disk
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install dependencies and Git
RUN apt-get update && \
    apt-get install -y build-essential git && \
    apt-get install -y libgdal-dev gdal-bin && \
    apt-get install -y proj-bin proj-data && \
    apt-get clean

# Set the working directory inside the Docker container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port 5000 for Flask
EXPOSE 5000

# Run the application with Gunicorn for production
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
