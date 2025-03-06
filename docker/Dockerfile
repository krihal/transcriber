FROM debian:bookworm-slim

# Install required packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    python3-venv \
    git \
    nginx \
    supervisor


# Clone the repository
RUN git clone https://github.com/krihal/transcriber

# Set the working directory
WORKDIR ttt

# Create the virtual environment
RUN python3 -m venv venv

# Activate the virtual environment
RUN . venv/bin/activate

# Install the required packages
RUN pip3 install -r ttt/requirements.txt

# Set up the configuration files
COPY nginx.conf /etc/nginx/sites-available/default
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Start the services
CMD ["/usr/bin/supervisord"]

# Expose the port
EXPOSE 80
