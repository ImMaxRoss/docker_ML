# Use the official Python 3.11 image as the base image
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Copy all files from the 'backend' directory to the current working directory in the container
COPY . .

# Install dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Command to run the application
CMD ["python", "./cleaner.py"]