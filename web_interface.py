#!/usr/bin/env python3
"""
Web Interface for AI Agent System
A beautiful, modern web interface for interacting with the AI agent system
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, Any, List
import logging
from datetime import datetime

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from ai_agent_system import AIAgentSystem

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AI Agent System",
    description="Create perfect web applications with AI",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Global agent system instance
agent_system = None

@app.on_event("startup")
async def startup_event():
    """Initialize the AI agent system on startup"""
    global agent_system
    agent_system = AIAgentSystem()
    logger.info("AI Agent System web interface started")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with task input form"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/create-project")
async def create_project(request: Request, task_description: str = Form(...)):
    """Create a new project from user input"""
    try:
        global agent_system

        if not agent_system:
            raise HTTPException(status_code=500, detail="Agent system not initialized")

        # Create project
        project_config = await agent_system.create_project(task_description)

        # Start project execution in background
        asyncio.create_task(agent_system.execute_project(project_config))

        return JSONResponse({
            "success": True,
            "project_id": project_config.name,
            "message": f"Project '{project_config.name}' creation started!"
        })

    except Exception as e:
        logger.error(f"Error creating project: {str(e)}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.get("/status/{project_id}")
async def get_project_status(project_id: str):
    """Get project status"""
    try:
        global agent_system

        if not agent_system:
            raise HTTPException(status_code=500, detail="Agent system not initialized")

        status = agent_system.get_project_status()

        return JSONResponse({
            "success": True,
            "status": status
        })

    except Exception as e:
        logger.error(f"Error getting project status: {str(e)}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.get("/projects")
async def list_projects():
    """List all projects"""
    try:
        projects_path = Path("/workspace/projects")
        projects = []

        if projects_path.exists():
            for project_dir in projects_path.iterdir():
                if project_dir.is_dir():
                    config_file = project_dir / "project_config.json"
                    if config_file.exists():
                        with open(config_file, 'r') as f:
                            config = json.load(f)
                        projects.append({
                            "id": config["name"],
                            "name": config["name"],
                            "description": config["description"][:100] + "...",
                            "type": config["type"],
                            "created_at": config.get("created_at", "Unknown")
                        })

        return JSONResponse({
            "success": True,
            "projects": projects
        })

    except Exception as e:
        logger.error(f"Error listing projects: {str(e)}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.get("/examples")
async def get_examples():
    """Get example project descriptions"""
    examples = [
        {
            "title": "E-commerce Platform",
            "description": "Create a modern e-commerce application with React frontend and Node.js backend. Features needed: user authentication, product catalog, shopping cart, checkout process, payment integration with Stripe, order management, admin dashboard, responsive design, search and filtering, user reviews and ratings, wishlist functionality. Use PostgreSQL database, include email notifications, and make it mobile-friendly."
        },
        {
            "title": "Social Media Dashboard",
            "description": "Build a social media analytics dashboard with Vue.js frontend and Python FastAPI backend. Include real-time data visualization, user management, post analytics, engagement metrics, multi-platform integration (Twitter, Facebook, Instagram), reporting features, and team collaboration tools."
        },
        {
            "title": "Task Management App",
            "description": "Create a project management application with Next.js frontend and Django backend. Features: task creation and assignment, progress tracking, team collaboration, file attachments, time tracking, deadline management, notifications, and mobile-responsive design."
        },
        {
            "title": "Blog Platform",
            "description": "Develop a full-featured blog platform with Svelte frontend and Flask backend. Include article creation and editing, user authentication, commenting system, categories and tags, SEO optimization, RSS feeds, newsletter integration, and admin panel."
        },
        {
            "title": "Learning Management System",
            "description": "Build an educational platform with Angular frontend and Node.js backend. Features: course creation, video streaming, quizzes and assessments, student progress tracking, certificates, discussion forums, and payment integration for premium content."
        }
    ]

    return JSONResponse({
        "success": True,
        "examples": examples
    })

def create_static_files():
    """Create static files and templates for the web interface"""

    # Create static directory
    static_dir = Path("static")
    static_dir.mkdir(exist_ok=True)

    # Create CSS file
    css_content = '''
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
}

.container {
    max-width: 800px;
    width: 100%;
    padding: 2rem;
}

.card {
    background: white;
    border-radius: 20px;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.1);
    overflow: hidden;
    margin-bottom: 2rem;
}

.header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 3rem 2rem;
    text-align: center;
}

.header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 1rem;
}

.header p {
    font-size: 1.2rem;
    opacity: 0.9;
}

.form-section {
    padding: 3rem 2rem;
}

.form-group {
    margin-bottom: 2rem;
}

.form-group label {
    display: block;
    font-weight: 600;
    margin-bottom: 0.5rem;
    color: #333;
}

textarea {
    width: 100%;
    padding: 1rem;
    border: 2px solid #e1e5e9;
    border-radius: 10px;
    font-size: 1rem;
    font-family: inherit;
    resize: vertical;
    min-height: 150px;
    transition: border-color 0.3s ease;
}

textarea:focus {
    outline: none;
    border-color: #667eea;
}

.btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    padding: 1rem 2rem;
    font-size: 1.1rem;
    font-weight: 600;
    border-radius: 10px;
    cursor: pointer;
    transition: all 0.3s ease;
    width: 100%;
}

.btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
}

.btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
}

.examples-section {
    background: #f8f9fa;
    padding: 2rem;
    border-radius: 15px;
}

.examples-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1rem;
    margin-top: 1rem;
}

.example-card {
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    border: 1px solid #e1e5e9;
    cursor: pointer;
    transition: all 0.3s ease;
}

.example-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
}

.example-card h3 {
    color: #667eea;
    margin-bottom: 0.5rem;
}

.example-card p {
    color: #666;
    font-size: 0.9rem;
    line-height: 1.5;
}

.status-section {
    margin-top: 2rem;
    padding: 2rem;
    background: white;
    border-radius: 15px;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
}

.status-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-top: 1rem;
}

.status-card {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
}

.status-card .number {
    font-size: 2rem;
    font-weight: 700;
    color: #667eea;
}

.status-card .label {
    color: #666;
    font-size: 0.9rem;
}

.progress-bar {
    width: 100%;
    height: 10px;
    background: #e1e5e9;
    border-radius: 5px;
    overflow: hidden;
    margin: 1rem 0;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    transition: width 0.3s ease;
}

.loading {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid rgba(255,255,255,.3);
    border-radius: 50%;
    border-top-color: #fff;
    animation: spin 1s ease-in-out infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

.success {
    background: #d4edda;
    color: #155724;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}

.error {
    background: #f8d7da;
    color: #721c24;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}

@media (max-width: 768px) {
    .container {
        padding: 1rem;
    }

    .header h1 {
        font-size: 2rem;
    }

    .header p {
        font-size: 1rem;
    }

    .form-section {
        padding: 2rem 1rem;
    }

    .examples-grid {
        grid-template-columns: 1fr;
    }
}
'''
    with open(static_dir / "styles.css", 'w') as f:
        f.write(css_content)

    # Create JavaScript file
    js_content = '''
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('project-form');
    const submitBtn = document.getElementById('submit-btn');
    const statusSection = document.getElementById('status-section');
    const progressBar = document.getElementById('progress-bar');
    const progressFill = document.getElementById('progress-fill');
    const examplesGrid = document.getElementById('examples-grid');

    // Handle form submission
    form.addEventListener('submit', async function(e) {
        e.preventDefault();

        const formData = new FormData(form);
        const taskDescription = formData.get('task_description');

        if (!taskDescription.trim()) {
            alert('Please describe your project');
            return;
        }

        // Show loading state
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<span class="loading"></span> Creating Project...';

        try {
            const response = await fetch('/create-project', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.success) {
                showSuccess(result.message);
                startStatusPolling(result.project_id);
            } else {
                showError(result.error);
            }
        } catch (error) {
            showError('Failed to create project. Please try again.');
            console.error('Error:', error);
        } finally {
            // Reset button state
            submitBtn.disabled = false;
            submitBtn.innerHTML = 'Create Project';
        }
    });

    // Handle example selection
    if (examplesGrid) {
        examplesGrid.addEventListener('click', function(e) {
            if (e.target.closest('.example-card')) {
                const card = e.target.closest('.example-card');
                const description = card.querySelector('p').textContent;
                document.getElementById('task_description').value = description;
            }
        });
    }

    // Status polling
    let statusInterval;

    function startStatusPolling(projectId) {
        statusSection.style.display = 'block';

        statusInterval = setInterval(async () => {
            try {
                const response = await fetch(`/status/${projectId}`);
                const result = await response.json();

                if (result.success) {
                    const status = result.status;
                    updateStatusDisplay(status);

                    if (status.status === 'completed' || status.status === 'failed') {
                        clearInterval(statusInterval);
                    }
                }
            } catch (error) {
                console.error('Error polling status:', error);
            }
        }, 2000);
    }

    function updateStatusDisplay(status) {
        // Update progress bar
        const progress = status.progress_percentage || 0;
        progressFill.style.width = progress + '%';

        // Update status cards
        document.getElementById('total-tasks').textContent = status.total_tasks || 0;
        document.getElementById('completed-tasks').textContent = status.completed_tasks || 0;
        document.getElementById('pending-tasks').textContent = status.pending_tasks || 0;
        document.getElementById('in-progress-tasks').textContent = status.in_progress_tasks || 0;

        // Update project name
        if (status.project_name) {
            document.getElementById('project-name').textContent = status.project_name;
        }
    }

    function showSuccess(message) {
        const alert = document.createElement('div');
        alert.className = 'success';
        alert.textContent = message;
        statusSection.insertBefore(alert, statusSection.firstChild);
        setTimeout(() => alert.remove(), 5000);
    }

    function showError(message) {
        const alert = document.createElement('div');
        alert.className = 'error';
        alert.textContent = message;
        statusSection.insertBefore(alert, statusSection.firstChild);
        setTimeout(() => alert.remove(), 5000);
    }

    // Load examples on page load
    loadExamples();
});

async function loadExamples() {
    try {
        const response = await fetch('/examples');
        const result = await response.json();

        if (result.success) {
            const examplesGrid = document.getElementById('examples-grid');
            examplesGrid.innerHTML = '';

            result.examples.forEach(example => {
                const card = document.createElement('div');
                card.className = 'example-card';
                card.innerHTML = `
                    <h3>${example.title}</h3>
                    <p>${example.description}</p>
                `;
                examplesGrid.appendChild(card);
            });
        }
    } catch (error) {
        console.error('Error loading examples:', error);
    }
}
'''
    with open(static_dir / "script.js", 'w') as f:
        f.write(js_content)

    # Create templates directory
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)

    # Create index.html template
    html_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Agent System - Create Perfect Web Applications</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="/static/styles.css">
</head>
<body>
    <div class="container">
        <div class="card">
            <div class="header">
                <h1>🤖 AI Agent System</h1>
                <p>Create perfect web applications with artificial intelligence</p>
            </div>

            <div class="form-section">
                <form id="project-form">
                    <div class="form-group">
                        <label for="task_description">Describe your project:</label>
                        <textarea
                            id="task_description"
                            name="task_description"
                            placeholder="Example: Create a modern e-commerce application with React frontend and Node.js backend. Features needed: user authentication, product catalog, shopping cart, checkout process, payment integration with Stripe, order management, admin dashboard, responsive design..."
                            required
                        ></textarea>
                    </div>

                    <button type="submit" class="btn" id="submit-btn">
                        Create Project
                    </button>
                </form>
            </div>
        </div>

        <div class="card examples-section">
            <h2 style="margin-bottom: 1rem;">📝 Example Projects</h2>
            <p style="margin-bottom: 1rem;">Click on any example to get started quickly:</p>
            <div class="examples-grid" id="examples-grid">
                <!-- Examples will be loaded here -->
            </div>
        </div>

        <div class="card status-section" id="status-section" style="display: none;">
            <h2>📊 Project Status</h2>
            <div id="status-alerts"></div>

            <div class="status-grid">
                <div class="status-card">
                    <div class="number" id="total-tasks">0</div>
                    <div class="label">Total Tasks</div>
                </div>
                <div class="status-card">
                    <div class="number" id="completed-tasks">0</div>
                    <div class="label">Completed</div>
                </div>
                <div class="status-card">
                    <div class="number" id="pending-tasks">0</div>
                    <div class="label">Pending</div>
                </div>
                <div class="status-card">
                    <div class="number" id="in-progress-tasks">0</div>
                    <div class="label">In Progress</div>
                </div>
            </div>

            <div class="progress-bar">
                <div class="progress-fill" id="progress-fill" style="width: 0%;"></div>
            </div>

            <p id="project-name" style="text-align: center; font-weight: 600;"></p>
        </div>
    </div>

    <script src="/static/script.js"></script>
</body>
</html>
'''
    with open(templates_dir / "index.html", 'w') as f:
        f.write(html_content)

if __name__ == "__main__":
    # Create static files and templates
    create_static_files()

    # Start the web server
    uvicorn.run(app, host="0.0.0.0", port=8000)