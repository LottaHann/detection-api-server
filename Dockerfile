# Use Python 3.10 as the base image
FROM python:3.10-slim

ARG DEFAULT_CAMERA=opencv
ARG DEFAULT_MODULE=0

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt


# Make port 80 available to the world outside this container
# Adjust this if your application uses a different port
EXPOSE 8008

# Define environment variable
ENV CAMERA=${DEFAULT_CAMERA}
ENV MODULE=${DEFAULT_MODULE}
# Run app.py when the container launches
CMD ["python", "app.py"]