# Use the official Python image as a base
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the requirements file into the container
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port that the Flask app runs on
EXPOSE 5006

# Set the command to run the Flask app
CMD ["flask", "run", "--host=0.0.0.0", "--port=5006"]
