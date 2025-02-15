# Use official Python image
FROM python:3.8

# Set the working directory inside the container
WORKDIR /app

# Copy files
COPY requirements.txt requirements.txt
COPY app/ app/
COPY artifacts/ artifacts/
COPY config/ config/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for API
EXPOSE 5000

# Command to run the API
CMD ["python", "app/app.py"]
