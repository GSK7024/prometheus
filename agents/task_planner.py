"""
Task Planner Agent
Responsible for breaking down complex projects into specific, actionable tasks
with proper dependencies and priority ordering.
"""

import uuid
import json
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class ProjectTask:
    """Represents a specific task for project execution"""
    id: str
    description: str
    category: str  # 'setup', 'frontend', 'backend', 'database', 'testing', 'deployment'
    priority: int  # 1-10, higher number = higher priority
    estimated_time: int  # minutes
    dependencies: List[str]  # list of task IDs
    required_skills: List[str]
    deliverables: List[str]
    complexity: str  # 'low', 'medium', 'high'

class TaskPlanner:
    """
    Advanced task planning system that creates detailed execution plans
    for complex web applications with proper dependency management.
    """

    def __init__(self, agent_system):
        self.agent_system = agent_system
        self.task_templates = self._load_task_templates()

    def _load_task_templates(self) -> Dict[str, Any]:
        """Load task templates for different project types"""
        templates = {
            'react_setup': {
                'description': 'Set up React application with modern toolchain',
                'category': 'setup',
                'priority': 10,
                'estimated_time': 15,
                'required_skills': ['frontend'],
                'deliverables': ['package.json', 'src/App.js', 'public/index.html']
            },
            'database_setup': {
                'description': 'Set up database and connection',
                'category': 'database',
                'priority': 9,
                'estimated_time': 20,
                'required_skills': ['database'],
                'deliverables': ['database config', 'connection files']
            },
            'authentication': {
                'description': 'Implement user authentication system',
                'category': 'backend',
                'priority': 8,
                'estimated_time': 45,
                'required_skills': ['backend', 'frontend'],
                'deliverables': ['auth routes', 'login components', 'JWT handling']
            },
            'ui_components': {
                'description': 'Create reusable UI components',
                'category': 'frontend',
                'priority': 7,
                'estimated_time': 30,
                'required_skills': ['frontend', 'ui_ux'],
                'deliverables': ['component library', 'styling files']
            },
            'api_endpoints': {
                'description': 'Build REST API endpoints',
                'category': 'backend',
                'priority': 6,
                'estimated_time': 40,
                'required_skills': ['backend'],
                'deliverables': ['API routes', 'controllers', 'middleware']
            }
        }
        return templates

    async def create_project_tasks(self, project_config) -> List[ProjectTask]:
        """
        Create a comprehensive task list for a project
        Analyzes requirements and creates detailed execution plan
        """
        tasks = []

        # Base setup tasks (always required)
        setup_tasks = await self._create_setup_tasks(project_config)
        tasks.extend(setup_tasks)

        # Feature-specific tasks
        feature_tasks = await self._create_feature_tasks(project_config)
        tasks.extend(feature_tasks)

        # Testing tasks
        testing_tasks = await self._create_testing_tasks(project_config)
        tasks.extend(testing_tasks)

        # Deployment tasks
        deployment_tasks = await self._create_deployment_tasks(project_config)
        tasks.extend(deployment_tasks)

        # Resolve dependencies and optimize order
        tasks = self._optimize_task_order(tasks)

        logger.info(f"Created {len(tasks)} tasks for project {project_config.name}")
        return tasks

    async def _create_setup_tasks(self, project_config) -> List[ProjectTask]:
        """Create setup tasks based on project type"""
        tasks = []

        # Project directory setup
        tasks.append(ProjectTask(
            id=str(uuid.uuid4()),
            description=f"Create project directory structure for {project_config.name}",
            category='setup',
            priority=10,
            estimated_time=5,
            dependencies=[],
            required_skills=['file_management'],
            deliverables=['project directory', 'basic folder structure'],
            complexity='low'
        ))

        # Frontend setup
        if project_config.frontend_framework:
            tasks.append(ProjectTask(
                id=str(uuid.uuid4()),
                description=f"Initialize {project_config.frontend_framework} frontend application",
                category='setup',
                priority=10,
                estimated_time=15,
                dependencies=[],
                required_skills=['frontend'],
                deliverables=[f'{project_config.frontend_framework} app', 'package.json', 'basic components'],
                complexity='medium'
            ))

        # Backend setup
        if project_config.backend_framework:
            tasks.append(ProjectTask(
                id=str(uuid.uuid4()),
                description=f"Initialize {project_config.backend_framework} backend application",
                category='setup',
                priority=10,
                estimated_time=15,
                dependencies=[],
                required_skills=['backend'],
                deliverables=[f'{project_config.backend_framework} server', 'package.json', 'basic routes'],
                complexity='medium'
            ))

        # Database setup
        if project_config.database:
            tasks.append(ProjectTask(
                id=str(uuid.uuid4()),
                description=f"Set up {project_config.database} database and connection",
                category='database',
                priority=9,
                estimated_time=20,
                dependencies=[],
                required_skills=['database'],
                deliverables=['database config', 'connection files', 'schema'],
                complexity='medium'
            ))

        return tasks

    async def _create_feature_tasks(self, project_config) -> List[ProjectTask]:
        """Create tasks based on required features"""
        tasks = []
        features = project_config.features

        # Authentication system
        if 'authentication' in features or 'login' in features:
            tasks.extend(await self._create_authentication_tasks())

        # E-commerce features
        if 'e-commerce' in features or 'payment' in features:
            tasks.extend(await self._create_ecommerce_tasks())

        # Dashboard
        if 'dashboard' in features:
            tasks.extend(await self._create_dashboard_tasks())

        # API features
        if 'api' in features:
            tasks.extend(await self._create_api_tasks())

        # Real-time features
        if 'real-time' in features or 'websocket' in features:
            tasks.extend(await self._create_realtime_tasks())

        return tasks

    async def _create_authentication_tasks(self) -> List[ProjectTask]:
        """Create authentication-related tasks"""
        return [
            ProjectTask(
                id=str(uuid.uuid4()),
                description='Implement user registration and login system',
                category='backend',
                priority=8,
                estimated_time=45,
                dependencies=[],
                required_skills=['backend', 'database'],
                deliverables=['auth routes', 'user model', 'JWT tokens'],
                complexity='high'
            ),
            ProjectTask(
                id=str(uuid.uuid4()),
                description='Create login and registration UI components',
                category='frontend',
                priority=8,
                estimated_time=30,
                dependencies=[],
                required_skills=['frontend', 'ui_ux'],
                deliverables=['login form', 'registration form', 'auth guards'],
                complexity='medium'
            ),
            ProjectTask(
                id=str(uuid.uuid4()),
                description='Implement password reset functionality',
                category='backend',
                priority=7,
                estimated_time=25,
                dependencies=[],
                required_skills=['backend', 'email'],
                deliverables=['password reset routes', 'email templates'],
                complexity='medium'
            )
        ]

    async def _create_ecommerce_tasks(self) -> List[ProjectTask]:
        """Create e-commerce related tasks"""
        return [
            ProjectTask(
                id=str(uuid.uuid4()),
                description='Create product catalog and management system',
                category='fullstack',
                priority=7,
                estimated_time=60,
                dependencies=[],
                required_skills=['backend', 'frontend', 'database'],
                deliverables=['product models', 'catalog API', 'product components'],
                complexity='high'
            ),
            ProjectTask(
                id=str(uuid.uuid4()),
                description='Implement shopping cart functionality',
                category='frontend',
                priority=6,
                estimated_time=35,
                dependencies=[],
                required_skills=['frontend'],
                deliverables=['cart components', 'cart state management', 'local storage'],
                complexity='medium'
            ),
            ProjectTask(
                id=str(uuid.uuid4()),
                description='Build checkout and payment processing',
                category='fullstack',
                priority=5,
                estimated_time=50,
                dependencies=[],
                required_skills=['backend', 'frontend', 'payment'],
                deliverables=['checkout flow', 'payment integration', 'order processing'],
                complexity='high'
            )
        ]

    async def _create_dashboard_tasks(self) -> List[ProjectTask]:
        """Create dashboard-related tasks"""
        return [
            ProjectTask(
                id=str(uuid.uuid4()),
                description='Create admin dashboard layout and navigation',
                category='frontend',
                priority=6,
                estimated_time=40,
                dependencies=[],
                required_skills=['frontend', 'ui_ux'],
                deliverables=['dashboard layout', 'navigation components', 'responsive design'],
                complexity='medium'
            ),
            ProjectTask(
                id=str(uuid.uuid4()),
                description='Implement dashboard widgets and charts',
                category='frontend',
                priority=5,
                estimated_time=30,
                dependencies=[],
                required_skills=['frontend'],
                deliverables=['data visualization', 'chart components', 'dashboard widgets'],
                complexity='medium'
            )
        ]

    async def _create_api_tasks(self) -> List[ProjectTask]:
        """Create API-related tasks"""
        return [
            ProjectTask(
                id=str(uuid.uuid4()),
                description='Design and implement REST API architecture',
                category='backend',
                priority=7,
                estimated_time=40,
                dependencies=[],
                required_skills=['backend'],
                deliverables=['API routes', 'controllers', 'middleware', 'documentation'],
                complexity='high'
            ),
            ProjectTask(
                id=str(uuid.uuid4()),
                description='Implement API security and rate limiting',
                category='backend',
                priority=6,
                estimated_time=25,
                dependencies=[],
                required_skills=['backend'],
                deliverables=['authentication middleware', 'rate limiting', 'security headers'],
                complexity='medium'
            )
        ]

    async def _create_realtime_tasks(self) -> List[ProjectTask]:
        """Create real-time functionality tasks"""
        return [
            ProjectTask(
                id=str(uuid.uuid4()),
                description='Set up WebSocket server and client connections',
                category='fullstack',
                priority=5,
                estimated_time=35,
                dependencies=[],
                required_skills=['backend', 'frontend'],
                deliverables=['WebSocket server', 'client connections', 'real-time updates'],
                complexity='medium'
            )
        ]

    async def _create_testing_tasks(self, project_config) -> List[ProjectTask]:
        """Create testing-related tasks"""
        return [
            ProjectTask(
                id=str(uuid.uuid4()),
                description='Set up testing framework and configuration',
                category='testing',
                priority=4,
                estimated_time=20,
                dependencies=[],
                required_skills=['testing'],
                deliverables=['test configuration', 'test scripts', 'CI/CD setup'],
                complexity='medium'
            ),
            ProjectTask(
                id=str(uuid.uuid4()),
                description='Write unit tests for components and services',
                category='testing',
                priority=3,
                estimated_time=45,
                dependencies=[],
                required_skills=['testing'],
                deliverables=['unit tests', 'test coverage reports'],
                complexity='medium'
            )
        ]

    async def _create_deployment_tasks(self, project_config) -> List[ProjectTask]:
        """Create deployment-related tasks"""
        return [
            ProjectTask(
                id=str(uuid.uuid4()),
                description='Configure production deployment settings',
                category='deployment',
                priority=2,
                estimated_time=30,
                dependencies=[],
                required_skills=['deployment'],
                deliverables=['deployment config', 'environment variables', 'Docker files'],
                complexity='medium'
            ),
            ProjectTask(
                id=str(uuid.uuid4()),
                description='Set up monitoring and logging',
                category='deployment',
                priority=1,
                estimated_time=20,
                dependencies=[],
                required_skills=['deployment'],
                deliverables=['monitoring setup', 'logging configuration', 'error tracking'],
                complexity='low'
            )
        ]

    def _optimize_task_order(self, tasks: List[ProjectTask]) -> List[ProjectTask]:
        """
        Optimize task execution order based on dependencies and priorities
        Uses topological sorting with priority consideration
        """
        # Sort by priority (higher first), then by dependencies
        sorted_tasks = sorted(tasks, key=lambda x: (-x.priority, len(x.dependencies)))

        # Resolve dependencies - ensure no circular dependencies
        ordered_tasks = []
        processed = set()

        def can_execute(task):
            return all(dep in processed for dep in task.dependencies)

        while len(ordered_tasks) < len(tasks):
            # Find tasks that can be executed
            executable = [t for t in sorted_tasks if t.id not in processed and can_execute(t)]

            if not executable:
                # Handle circular dependencies or missing dependencies
                logger.warning("Circular dependency detected or missing dependencies")
                remaining = [t for t in sorted_tasks if t.id not in processed]
                executable = remaining[:1]  # Take first remaining task

            # Execute highest priority task
            task = executable[0]
            ordered_tasks.append(task)
            processed.add(task.id)

        return ordered_tasks

    def estimate_project_time(self, tasks: List[ProjectTask]) -> Dict[str, Any]:
        """Estimate total project completion time"""
        total_time = sum(task.estimated_time for task in tasks)
        categories = {}

        for task in tasks:
            if task.category not in categories:
                categories[task.category] = 0
            categories[task.category] += task.estimated_time

        return {
            'total_minutes': total_time,
            'total_hours': total_time / 60,
            'by_category': categories,
            'parallel_potential': len([t for t in tasks if not t.dependencies])
        }