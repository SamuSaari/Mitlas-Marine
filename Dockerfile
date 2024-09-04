# Use the official Python image from Docker Hub
FROM python:3.9-slim

# Set environment variables (new format)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the Docker container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port 5000 for Flask
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
