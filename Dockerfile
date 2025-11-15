# Use a slim Python image for a smaller final size
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
# This speeds up subsequent builds if requirements.txt doesn't change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and the environment file
COPY app /app/app
COPY .env /app/.env

# Expose the port the app runs on
EXPOSE 8000

WORKDIR /app/app

# Command to run the application using Uvicorn
# We use the Gunicorn worker class for better production performance
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]