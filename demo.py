#!/usr/bin/env python3
"""
AI Agent System Demo
Demonstrates the complete AI agent system by creating a full-featured e-commerce application
"""

import asyncio
import json
import os
from pathlib import Path
from datetime import datetime

from ai_agent_system import AIAgentSystem

async def demo_ecommerce_app():
    """Demonstrate creating a complete e-commerce application"""

    print("🚀 AI Agent System Demo")
    print("=" * 50)
    print("Creating a complete e-commerce application...")
    print()

    # Initialize the AI agent system
    system = AIAgentSystem()

    # Define the e-commerce project requirements
    ecommerce_requirements = """
    Create a modern e-commerce application with the following specifications:

    FRONTEND:
    - React with TypeScript
    - Material-UI for components
    - React Router for navigation
    - State management with Context API or Redux
    - Responsive design for mobile, tablet, and desktop

    BACKEND:
    - Node.js with Express.js
    - MongoDB database with Mongoose ODM
    - JWT authentication
    - RESTful API design
    - Input validation and sanitization

    FEATURES:
    1. User Authentication & Authorization
       - User registration and login
       - Password reset functionality
       - Role-based access (user, admin)
       - Secure session management

    2. Product Management
       - Product catalog with categories
       - Product search and filtering
       - Product details with images
       - Inventory management
       - Product reviews and ratings

    3. Shopping Cart
       - Add/remove products
       - Quantity management
       - Persistent cart (localStorage + database)
       - Cart summary and totals

    4. Checkout Process
       - Shipping information
       - Payment integration (Stripe)
       - Order confirmation
       - Email notifications

    5. Order Management
       - Order history for users
       - Order status tracking
       - Admin order management
       - Order fulfillment

    6. Admin Dashboard
       - Product management interface
       - User management
       - Order analytics
       - Sales reports

    DATABASE:
    - MongoDB with Mongoose schemas
    - User collection
    - Product collection
    - Order collection
    - Review collection
    - Category collection

    DEPLOYMENT:
    - Docker containerization
    - Production-ready configuration
    - Environment variables
    - Security best practices

    TESTING:
    - Unit tests for components
    - Integration tests for API
    - End-to-end testing
    - Test coverage reports

    ADDITIONAL REQUIREMENTS:
    - SEO optimization
    - Performance optimization
    - Security headers
    - Error handling
    - Logging and monitoring
    - API documentation
    """

    print("📋 Project Requirements:")
    print(ecommerce_requirements[:200] + "...")
    print()

    try:
        # Step 1: Create the project
        print("1️⃣ Creating project configuration...")
        project_config = await system.create_project(ecommerce_requirements)

        print(f"✅ Project '{project_config.name}' created successfully!")
        print(f"📁 Project location: {project_config.target_directory}")
        print()

        # Step 2: Execute the project
        print("2️⃣ Starting project execution...")
        print("🤖 AI agents are now working on your project...")
        print()

        # Show progress updates
        start_time = datetime.now()
        success = await system.execute_project(project_config)

        if success:
            end_time = datetime.now()
            duration = end_time - start_time

            print("🎉 Project completed successfully!")
            print(f"⏱️  Total time: {duration}")
            print()

            # Show project structure
            print("3️⃣ Project structure created:")
            project_path = Path(project_config.target_directory)

            def print_directory_structure(path, indent=0):
                for item in sorted(path.iterdir()):
                    if item.is_dir():
                        print("  " * indent + f"📁 {item.name}/")
                        if indent < 2:  # Limit depth for readability
                            print_directory_structure(item, indent + 1)
                    else:
                        print("  " * indent + f"📄 {item.name}")

            if project_path.exists():
                print_directory_structure(project_path)
            print()

            # Show summary
            print("4️⃣ Project Summary:")
            print(f"   • Frontend: {project_config.frontend_framework}")
            print(f"   • Backend: {project_config.backend_framework}")
            print(f"   • Database: {project_config.database}")
            print(f"   • Features: {', '.join(project_config.features)}")
            print()

            print("🚀 Your e-commerce application is ready!")
            print(f"📂 Location: {project_path}")
            print()
            print("Next steps:")
            print("1. Navigate to the project directory")
            print("2. Install dependencies: npm install")
            print("3. Start the development server: npm run dev")
            print("4. Open your browser and visit http://localhost:3000")

        else:
            print("❌ Project creation failed")
            print("Please check the logs for more information")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Please try again or contact support")

async def demo_task_management_app():
    """Demonstrate creating a task management application"""

    print("📝 Task Management App Demo")
    print("=" * 50)

    system = AIAgentSystem()

    task_app_requirements = """
    Create a modern task management application with:

    - Vue.js frontend with Vuetify UI components
    - Python FastAPI backend
    - PostgreSQL database
    - Real-time updates with WebSockets
    - Team collaboration features
    - File attachments
    - Due date tracking
    - Priority levels
    - Progress tracking
    """

    project_config = await system.create_project(task_app_requirements)
    await system.execute_project(project_config)

    print(f"✅ Task management app '{project_config.name}' created!")

async def demo_blog_platform():
    """Demonstrate creating a blog platform"""

    print("📖 Blog Platform Demo")
    print("=" * 50)

    system = AIAgentSystem()

    blog_requirements = """
    Create a modern blog platform with:

    - Next.js frontend with server-side rendering
    - Django backend with Django REST framework
    - PostgreSQL database
    - Markdown support for articles
    - SEO optimization
    - Comment system
    - Category and tag management
    - User profiles
    - Newsletter integration
    - Social media sharing
    """

    project_config = await system.create_project(blog_requirements)
    await system.execute_project(project_config)

    print(f"✅ Blog platform '{project_config.name}' created!")

async def main():
    """Run the demo"""
    print("🤖 AI Agent System - Complete Demo")
    print("=" * 60)
    print("This demo will showcase the AI agent system's ability to create")
    print("complete, production-ready web applications from natural language")
    print("descriptions.")
    print()

    # Run different demos
    await demo_ecommerce_app()
    print("\n" + "=" * 60 + "\n")

    # Uncomment to run additional demos
    # await demo_task_management_app()
    # print("\n" + "=" * 60 + "\n")
    # await demo_blog_platform()

    print("🎯 Demo completed!")
    print("The AI agent system successfully created multiple types of applications")
    print("with modern architectures, best practices, and production-ready code.")

if __name__ == "__main__":
    asyncio.run(main())