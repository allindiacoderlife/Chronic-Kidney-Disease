# Stage 1: Build the React frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /app/client
COPY client/package*.json ./
RUN npm ci
COPY client/ ./
# We set VITE_API_URL to an empty string so the frontend uses relative paths.
# This ensures it connects back to the same origin where it is hosted.
ENV VITE_API_URL=""
RUN npm run build

# Stage 2: Build the backend and serve the application
FROM python:3.11-slim
WORKDIR /app

# Install system dependencies, Nginx, and Supervisor
RUN apt-get update && apt-get install -y \
    nginx \
    supervisor \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY server/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the server application code
COPY server/ ./

# Copy the compiled React build from the frontend stage into Nginx
COPY --from=frontend-builder /app/client/dist /usr/share/nginx/html

# Copy deployment configurations
COPY nginx.conf /etc/nginx/sites-available/default
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Create necessary directories for logs
RUN mkdir -p logs /var/log/supervisor

# Expose port 80 for Render routing
EXPOSE 80

# Start Supervisor (which will run both Nginx and Gunicorn simultaneously)
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
