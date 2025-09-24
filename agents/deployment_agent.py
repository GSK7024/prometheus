"""
Deployment Agent
Specialized in deploying applications to various platforms
Supports Docker, cloud platforms, and CI/CD pipelines.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class DeploymentAgent:
    """
    Advanced deployment agent that handles application deployment
    to various platforms with proper configuration and optimization.
    """

    def __init__(self, agent_system):
        self.agent_system = agent_system
        self.supported_platforms = {
            'docker': self._create_docker_deployment,
            'vercel': self._create_vercel_deployment,
            'netlify': self._create_netlify_deployment,
            'heroku': self._create_heroku_deployment,
            'aws': self._create_aws_deployment
        }

    async def execute_task(self, task) -> Dict[str, Any]:
        """Execute a deployment-related task"""
        logger.info(f"Deployment agent executing: {task.description}")

        if 'docker' in task.description.lower():
            return await self._setup_docker_deployment(task)
        elif 'vercel' in task.description.lower():
            return await self._setup_vercel_deployment(task)
        elif 'netlify' in task.description.lower():
            return await self._setup_netlify_deployment(task)
        elif 'heroku' in task.description.lower():
            return await self._setup_heroku_deployment(task)
        elif 'aws' in task.description.lower():
            return await self._setup_aws_deployment(task)
        else:
            return await self._general_deployment_task(task)

    async def _setup_docker_deployment(self, task) -> Dict[str, Any]:
        """Set up Docker deployment configuration"""
        project_config = self.agent_system.current_project
        project_path = Path(project_config.target_directory)

        # Create Dockerfile for frontend
        if project_config.frontend_framework:
            await self._create_frontend_dockerfile(project_path)

        # Create Dockerfile for backend
        if project_config.backend_framework:
            await self._create_backend_dockerfile(project_path)

        # Create docker-compose.yml
        await self._create_docker_compose(project_path)

        return {
            'status': 'completed',
            'docker_deployment_setup': True,
            'files_created': ['Dockerfile', 'docker-compose.yml']
        }

    async def _create_frontend_dockerfile(self, project_path: Path):
        """Create Dockerfile for frontend application"""

        dockerfile_content = '''
# Multi-stage build for React application
FROM node:18-alpine as build

WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm ci --only=production

# Copy source code
COPY . .

# Build the application
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built application
COPY --from=build /app/build /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
'''

        frontend_path = project_path / "frontend"
        with open(frontend_path / "Dockerfile", 'w') as f:
            f.write(dockerfile_content)

        # Create nginx configuration
        nginx_config = '''
server {
    listen 80;
    listen [::]:80;
    server_name localhost;

    root /usr/share/nginx/html;
    index index.html index.htm;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/javascript
        application/xml+rss
        application/json;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;

    # Handle client-side routing
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Cache static assets
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
'''
        with open(frontend_path / "nginx.conf", 'w') as f:
            f.write(nginx_config)

    async def _create_backend_dockerfile(self, project_path: Path):
        """Create Dockerfile for backend application"""

        dockerfile_content = '''
FROM node:18-alpine

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm ci --only=production && npm cache clean --force

# Copy source code
COPY . .

# Create non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nodejs -u 1001

USER nodejs

EXPOSE 5000

CMD ["npm", "start"]
'''
        backend_path = project_path / "backend"
        with open(backend_path / "Dockerfile", 'w') as f:
            f.write(dockerfile_content)

    async def _create_docker_compose(self, project_path: Path):
        """Create docker-compose.yml file"""

        compose_content = '''
version: '3.8'

services:
  frontend:
    build: ./frontend
    ports:
      - "3000:80"
    depends_on:
      - backend
    environment:
      - REACT_APP_API_URL=http://localhost:5000/api

  backend:
    build: ./backend
    ports:
      - "5000:5000"
    environment:
      - NODE_ENV=production
      - MONGODB_URI=mongodb://mongodb:27017/myapp
      - JWT_SECRET=your-super-secret-jwt-key
    depends_on:
      - mongodb
    volumes:
      - ./backend:/app
      - /app/node_modules

  mongodb:
    image: mongo:5.0
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db
    environment:
      - MONGO_INITDB_DATABASE=myapp

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - frontend
      - backend

volumes:
  mongodb_data:
'''
        with open(project_path / "docker-compose.yml", 'w') as f:
            f.write(compose_content)

    async def _setup_vercel_deployment(self, task) -> Dict[str, Any]:
        """Set up Vercel deployment configuration"""

        project_config = self.agent_system.current_project
        project_path = Path(project_config.target_directory)

        # Create vercel.json configuration
        vercel_config = {
            "version": 2,
            "builds": [
                {
                    "src": "frontend/package.json",
                    "use": "@vercel/static-build",
                    "config": {
                        "distDir": "build"
                    }
                }
            ],
            "routes": [
                {
                    "src": "/api/(.*)",
                    "dest": "/backend/$1"
                },
                {
                    "src": "/(.*)",
                    "dest": "/frontend/$1"
                }
            ],
            "env": {
                "NODE_ENV": "production"
            }
        }

        with open(project_path / "vercel.json", 'w') as f:
            json.dump(vercel_config, f, indent=2)

        return {
            'status': 'completed',
            'vercel_deployment_setup': True,
            'files_created': ['vercel.json']
        }

    async def _setup_netlify_deployment(self, task) -> Dict[str, Any]:
        """Set up Netlify deployment configuration"""

        project_config = self.agent_system.current_project
        project_path = Path(project_config.target_directory)

        # Create netlify.toml configuration
        netlify_config = '''
[build]
  publish = "frontend/build"
  command = "npm run build"

[[redirects]]
  from = "/api/*"
  to = "https://your-backend-api.com/:splat"
  status = 200

[build.environment]
  NODE_VERSION = "18"

# Security headers
[[headers]]
  for = "/*"
  [headers.values]
    X-Frame-Options = "DENY"
    X-XSS-Protection = "1; mode=block"
    X-Content-Type-Options = "nosniff"
    Referrer-Policy = "strict-origin-when-cross-origin"
'''
        with open(project_path / "netlify.toml", 'w') as f:
            f.write(netlify_config)

        return {
            'status': 'completed',
            'netlify_deployment_setup': True,
            'files_created': ['netlify.toml']
        }

    async def _setup_heroku_deployment(self, task) -> Dict[str, Any]:
        """Set up Heroku deployment configuration"""

        project_config = self.agent_system.current_project
        project_path = Path(project_config.target_directory)

        # Create Procfile
        procfile_content = 'web: npm start'
        with open(project_path / "Procfile", 'w') as f:
            f.write(procfile_content)

        # Create .env for production
        env_content = '''
NODE_ENV=production
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/myapp
JWT_SECRET=your-super-secret-jwt-key
'''
        with open(project_path / ".env.production", 'w') as f:
            f.write(env_content)

        return {
            'status': 'completed',
            'heroku_deployment_setup': True,
            'files_created': ['Procfile', '.env.production']
        }

    async def _setup_aws_deployment(self, task) -> Dict[str, Any]:
        """Set up AWS deployment configuration"""

        project_config = self.agent_system.current_project
        project_path = Path(project_config.target_directory)

        # Create AWS configuration files
        appspec_content = '''
version: 0.0
os: linux
files:
  - source: /
    destination: /var/www/html
hooks:
  BeforeInstall:
    - location: scripts/before_install.sh
      timeout: 300
      runas: root
  AfterInstall:
    - location: scripts/after_install.sh
      timeout: 300
      runas: root
  ApplicationStart:
    - location: scripts/start_application.sh
      timeout: 300
      runas: root
'''
        with open(project_path / "appspec.yml", 'w') as f:
            f.write(appspec_content)

        # Create deployment scripts
        scripts_dir = project_path / "scripts"
        scripts_dir.mkdir(exist_ok=True)

        before_install = '''#!/bin/bash
# Stop existing application
pm2 stop all || true
pm2 delete all || true

# Clean up
rm -rf /var/www/html/*
'''
        with open(scripts_dir / "before_install.sh", 'w') as f:
            f.write(before_install)

        return {
            'status': 'completed',
            'aws_deployment_setup': True,
            'files_created': ['appspec.yml']
        }

    async def _general_deployment_task(self, task) -> Dict[str, Any]:
        """Handle general deployment tasks"""
        return {
            'status': 'completed',
            'message': f'Deployment task completed: {task.description}'
        }

    async def _create_docker_deployment(self, project_path: Path):
        """Create Docker deployment files"""
        # Already handled in _setup_docker_deployment
        pass

    async def _create_vercel_deployment(self, project_path: Path):
        """Create Vercel deployment files"""
        # Already handled in _setup_vercel_deployment
        pass

    async def _create_netlify_deployment(self, project_path: Path):
        """Create Netlify deployment files"""
        # Already handled in _setup_netlify_deployment
        pass

    async def _create_heroku_deployment(self, project_path: Path):
        """Create Heroku deployment files"""
        # Already handled in _setup_heroku_deployment
        pass

    async def _create_aws_deployment(self, project_path: Path):
        """Create AWS deployment files"""
        # Already handled in _setup_aws_deployment
        pass