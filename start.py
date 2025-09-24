#!/usr/bin/env python3
"""
AI Agent System Startup Script
Easy way to start the AI agent system with various options
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path

def main():
    """Main startup function"""

    parser = argparse.ArgumentParser(
        description="AI Agent System - Create perfect web applications with AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start.py web              # Start web interface
  python start.py demo             # Run demo with e-commerce app
  python start.py demo --blog      # Run demo with blog platform
  python start.py demo --tasks     # Run demo with task management app
  python start.py api              # Start API server
  python start.py help             # Show this help message
        """
    )

    parser.add_argument(
        'command',
        choices=['web', 'demo', 'api', 'help'],
        help='Command to run'
    )

    parser.add_argument(
        '--blog',
        action='store_true',
        help='Run blog platform demo (only with demo command)'
    )

    parser.add_argument(
        '--tasks',
        action='store_true',
        help='Run task management demo (only with demo command)'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port for web interface (default: 8000)'
    )

    args = parser.parse_args()

    if args.command == 'help':
        parser.print_help()
        return

    elif args.command == 'web':
        print("🌐 Starting AI Agent System Web Interface...")
        print(f"📡 Server will be available at http://localhost:{args.port}")
        print("🔗 Open this URL in your browser to start creating applications")
        print()

        try:
            from web_interface import app
            import uvicorn

            uvicorn.run(app, host="0.0.0.0", port=args.port)
        except ImportError as e:
            print(f"❌ Error: Missing dependencies. Please install: {e}")
            print("Run: pip install fastapi uvicorn jinja2 python-multipart")
        except Exception as e:
            print(f"❌ Error starting web interface: {e}")

    elif args.command == 'demo':
        print("🚀 Running AI Agent System Demo...")

        if args.blog:
            print("📖 Creating Blog Platform Demo...")
            demo_type = "blog"
        elif args.tasks:
            print("📝 Creating Task Management Demo...")
            demo_type = "tasks"
        else:
            print("🛍️  Creating E-commerce Platform Demo...")
            demo_type = "ecommerce"

        try:
            from demo import demo_ecommerce_app, demo_task_management_app, demo_blog_platform

            if demo_type == "blog":
                asyncio.run(demo_blog_platform())
            elif demo_type == "tasks":
                asyncio.run(demo_task_management_app())
            else:
                asyncio.run(demo_ecommerce_app())

        except ImportError as e:
            print(f"❌ Error: Missing dependencies. Please install: {e}")
        except Exception as e:
            print(f"❌ Error running demo: {e}")

    elif args.command == 'api':
        print("🔌 Starting AI Agent System API...")
        print("This provides programmatic access to the AI agent system")

        try:
            from ai_agent_system import AIAgentSystem

            # Simple API server for the agent system
            import asyncio
            from fastapi import FastAPI
            import uvicorn

            api_app = FastAPI(title="AI Agent System API", version="1.0.0")

            @api_app.get("/")
            async def root():
                return {"message": "AI Agent System API", "status": "running"}

            @api_app.post("/create-project")
            async def create_project_endpoint(requirements: dict):
                system = AIAgentSystem()
                project_config = await system.create_project(requirements["description"])
                await system.execute_project(project_config)
                return {"status": "success", "project": project_config.name}

            print(f"📡 API server starting on port {args.port}")
            uvicorn.run(api_app, host="0.0.0.0", port=args.port)

        except ImportError as e:
            print(f"❌ Error: Missing dependencies. Please install: {e}")
        except Exception as e:
            print(f"❌ Error starting API: {e}")

def show_welcome():
    """Show welcome message"""
    print("🤖 AI Agent System - Create Perfect Web Applications")
    print("=" * 60)
    print("An advanced AI-powered system that creates complete, production-ready")
    print("web applications from natural language descriptions.")
    print()
    print("Choose a command to get started:")
    print("  web  - Start the web interface")
    print("  demo - Run a demonstration")
    print("  api  - Start the API server")
    print("  help - Show help information")
    print()
    print("For more information, see README.md")
    print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        show_welcome()
    else:
        main()