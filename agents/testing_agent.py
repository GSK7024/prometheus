"""
Testing Agent
Specialized in creating comprehensive test suites and ensuring code quality
Supports unit testing, integration testing, and end-to-end testing.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class TestingAgent:
    """
    Advanced testing agent that creates comprehensive test suites
    with unit tests, integration tests, and quality assurance checks.
    """

    def __init__(self, agent_system):
        self.agent_system = agent_system
        self.testing_frameworks = {
            'jest': self._setup_jest,
            'pytest': self._setup_pytest,
            'cypress': self._setup_cypress
        }

    async def execute_task(self, task) -> Dict[str, Any]:
        """Execute a testing-related task"""
        logger.info(f"Testing agent executing: {task.description}")

        if 'setup' in task.description.lower() and 'test' in task.description.lower():
            return await self._setup_testing_framework(task)
        elif 'unit' in task.description.lower():
            return await self._create_unit_tests(task)
        elif 'integration' in task.description.lower():
            return await self._create_integration_tests(task)
        elif 'e2e' in task.description.lower() or 'end-to-end' in task.description.lower():
            return await self._create_e2e_tests(task)
        else:
            return await self._general_testing_task(task)

    async def _setup_testing_framework(self, task) -> Dict[str, Any]:
        """Set up testing framework based on project type"""
        project_config = self.agent_system.current_project
        project_path = Path(project_config.target_directory)

        # Set up frontend testing
        if project_config.frontend_framework:
            await self._setup_frontend_testing(project_path)

        # Set up backend testing
        if project_config.backend_framework:
            await self._setup_backend_testing(project_path)

        return {
            'status': 'completed',
            'testing_framework_setup': True,
            'message': 'Testing framework configured'
        }

    async def _setup_frontend_testing(self, project_path: Path):
        """Set up frontend testing framework"""

        frontend_path = project_path / "frontend"

        # Create Jest configuration for React
        jest_config = {
            "testEnvironment": "jsdom",
            "setupFilesAfterEnv": ["<rootDir>/src/setupTests.js"],
            "testMatch": ["**/__tests__/**/*.(js,jsx,ts,tsx)"],
            "collectCoverageFrom": [
                "src/**/*.{js,jsx,ts,tsx}",
                "!src/index.js",
                "!src/setupTests.js"
            ],
            "coverageReporters": ["text", "lcov", "html"],
            "moduleNameMapping": {
                "\\.(css|less|scss|sass)$": "identity-obj-proxy"
            }
        }

        with open(frontend_path / "jest.config.js", 'w') as f:
            f.write(f"module.exports = {json.dumps(jest_config, indent=2)};\n")

        # Create setup file
        setup_content = '''
import '@testing-library/jest-dom';

// Mock IntersectionObserver
global.IntersectionObserver = class IntersectionObserver {
  constructor() {}

  disconnect() {}

  observe() {}

  unobserve() {}
};

// Mock ResizeObserver
global.ResizeObserver = class ResizeObserver {
  constructor() {}

  disconnect() {}

  observe() {}

  unobserve() {}
};
'''
        with open(frontend_path / "src" / "setupTests.js", 'w') as f:
            f.write(setup_content)

    async def _setup_backend_testing(self, project_path: Path):
        """Set up backend testing framework"""

        backend_path = project_path / "backend"

        # Set up Jest for Node.js
        if backend_path.exists():
            package_json = json.loads((backend_path / "package.json").read_text())

            # Add test scripts
            package_json["scripts"]["test"] = "jest"
            package_json["scripts"]["test:watch"] = "jest --watch"
            package_json["scripts"]["test:coverage"] = "jest --coverage"

            with open(backend_path / "package.json", 'w') as f:
                json.dump(package_json, f, indent=2)

    async def _create_unit_tests(self, task) -> Dict[str, Any]:
        """Create unit tests for components and functions"""
        project_config = self.agent_system.current_project
        project_path = Path(project_config.target_directory)

        # Create frontend unit tests
        if project_config.frontend_framework == 'react':
            await self._create_react_unit_tests(project_path)

        # Create backend unit tests
        if project_config.backend_framework == 'nodejs':
            await self._create_nodejs_unit_tests(project_path)

        return {
            'status': 'completed',
            'unit_tests_created': True,
            'message': 'Unit tests created'
        }

    async def _create_react_unit_tests(self, project_path: Path):
        """Create React component unit tests"""

        frontend_path = project_path / "frontend"

        # Test for Button component
        button_test = '''
import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import Button from '../src/components/Button';

describe('Button', () => {
  test('renders button with text', () => {
    render(<Button>Click me</Button>);
    const buttonElement = screen.getByText('Click me');
    expect(buttonElement).toBeInTheDocument();
  });

  test('calls onClick when clicked', () => {
    const handleClick = jest.fn();
    render(<Button onClick={handleClick}>Click me</Button>);
    const buttonElement = screen.getByText('Click me');
    fireEvent.click(buttonElement);
    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  test('is disabled when disabled prop is true', () => {
    render(<Button disabled>Click me</Button>);
    const buttonElement = screen.getByText('Click me');
    expect(buttonElement).toBeDisabled();
  });

  test('applies correct variant styles', () => {
    const { rerender } = render(<Button variant="primary">Primary</Button>);
    let buttonElement = screen.getByText('Primary');
    expect(buttonElement).toHaveClass('btn--primary');

    rerender(<Button variant="secondary">Secondary</Button>);
    buttonElement = screen.getByText('Secondary');
    expect(buttonElement).toHaveClass('btn--secondary');
  });
});
'''
        test_dir = frontend_path / "src" / "__tests__"
        test_dir.mkdir(exist_ok=True)
        with open(test_dir / "Button.test.js", 'w') as f:
            f.write(button_test)

        # Test for Dashboard component
        dashboard_test = '''
import React from 'react';
import { render, screen } from '@testing-library/react';
import Dashboard from '../pages/Dashboard';

describe('Dashboard', () => {
  test('renders dashboard title', () => {
    render(<Dashboard />);
    const titleElement = screen.getByText('Dashboard');
    expect(titleElement).toBeInTheDocument();
  });

  test('renders welcome message', () => {
    render(<Dashboard />);
    const welcomeMessage = screen.getByText(/Welcome to your modern dashboard/);
    expect(welcomeMessage).toBeInTheDocument();
  });
});
'''
        with open(test_dir / "Dashboard.test.js", 'w') as f:
            f.write(dashboard_test)

    async def _create_nodejs_unit_tests(self, project_path: Path):
        """Create Node.js unit tests"""

        backend_path = project_path / "backend"

        # Test for auth routes
        auth_test = '''
const request = require('supertest');
const app = require('../server');
const User = require('../models/User');

describe('Authentication Routes', () => {
  beforeEach(async () => {
    // Clear database before each test
    await User.deleteMany({});
  });

  describe('POST /api/auth/register', () => {
    test('should create a new user', async () => {
      const response = await request(app)
        .post('/api/auth/register')
        .send({
          name: 'Test User',
          email: 'test@example.com',
          password: 'password123'
        });

      expect(response.statusCode).toBe(201);
      expect(response.body.user.email).toBe('test@example.com');
      expect(response.body.token).toBeDefined();
    });

    test('should not create user with existing email', async () => {
      // Create user first
      await User.create({
        name: 'Existing User',
        email: 'existing@example.com',
        password: 'hashedpassword'
      });

      const response = await request(app)
        .post('/api/auth/register')
        .send({
          name: 'Test User',
          email: 'existing@example.com',
          password: 'password123'
        });

      expect(response.statusCode).toBe(400);
      expect(response.body.error).toBe('User already exists');
    });
  });

  describe('POST /api/auth/login', () => {
    test('should login with correct credentials', async () => {
      // Create user first
      const user = await User.create({
        name: 'Test User',
        email: 'test@example.com',
        password: await bcrypt.hash('password123', 10)
      });

      const response = await request(app)
        .post('/api/auth/login')
        .send({
          email: 'test@example.com',
          password: 'password123'
        });

      expect(response.statusCode).toBe(200);
      expect(response.body.token).toBeDefined();
    });

    test('should not login with incorrect password', async () => {
      const user = await User.create({
        name: 'Test User',
        email: 'test@example.com',
        password: await bcrypt.hash('password123', 10)
      });

      const response = await request(app)
        .post('/api/auth/login')
        .send({
          email: 'test@example.com',
          password: 'wrongpassword'
        });

      expect(response.statusCode).toBe(401);
      expect(response.body.error).toBe('Invalid credentials');
    });
  });
});
'''
        test_dir = backend_path / "__tests__"
        test_dir.mkdir(exist_ok=True)
        with open(test_dir / "auth.test.js", 'w') as f:
            f.write(auth_test)

    async def _create_integration_tests(self, task) -> Dict[str, Any]:
        """Create integration tests"""

        integration_test = '''
// Integration test example
describe('User Authentication Flow', () => {
  test('should register, login, and access protected route', async () => {
    // Register user
    const registerResponse = await request(app)
      .post('/api/auth/register')
      .send({
        name: 'Integration Test User',
        email: 'integration@example.com',
        password: 'password123'
      });

    expect(registerResponse.statusCode).toBe(201);

    // Login user
    const loginResponse = await request(app)
      .post('/api/auth/login')
      .send({
        email: 'integration@example.com',
        password: 'password123'
      });

    expect(loginResponse.statusCode).toBe(200);
    const token = loginResponse.body.token;

    // Access protected route
    const protectedResponse = await request(app)
      .get('/api/users/me')
      .set('Authorization', `Bearer ${token}`);

    expect(protectedResponse.statusCode).toBe(200);
    expect(protectedResponse.body.email).toBe('integration@example.com');
  });
});
'''

        project_path = Path(self.agent_system.current_project.target_directory)
        backend_path = project_path / "backend" / "__tests__"
        with open(backend_path / "integration.test.js", 'w') as f:
            f.write(integration_test)

        return {
            'status': 'completed',
            'integration_tests_created': True,
            'message': 'Integration tests created'
        }

    async def _create_e2e_tests(self, task) -> Dict[str, Any]:
        """Create end-to-end tests"""

        # Create Cypress configuration
        cypress_config = {
            "baseUrl": "http://localhost:3000",
            "viewportWidth": 1280,
            "viewportHeight": 720,
            "defaultCommandTimeout": 10000,
            "requestTimeout": 10000,
            "responseTimeout": 10000
        }

        project_path = Path(self.agent_system.current_project.target_directory)
        frontend_path = project_path / "frontend"

        with open(frontend_path / "cypress.config.js", 'w') as f:
            f.write(f"module.exports = {json.dumps(cypress_config, indent=2)}\n")

        # Create E2E test
        e2e_test = '''
describe('E-commerce App', () => {
  beforeEach(() => {
    cy.visit('/');
  });

  it('should load the homepage', () => {
    cy.contains('Dashboard').should('be.visible');
  });

  it('should allow user to login', () => {
    cy.visit('/login');
    cy.get('input[name="email"]').type('test@example.com');
    cy.get('input[name="password"]').type('password123');
    cy.get('button[type="submit"]').click();
    cy.url().should('include', '/');
  });

  it('should display products', () => {
    cy.visit('/products');
    cy.get('.product-card').should('have.length.at.least', 1);
  });
});
'''
        cypress_dir = frontend_path / "cypress" / "e2e"
        cypress_dir.mkdir(parents=True, exist_ok=True)
        with open(cypress_dir / "app.cy.js", 'w') as f:
            f.write(e2e_test)

        return {
            'status': 'completed',
            'e2e_tests_created': True,
            'message': 'End-to-end tests created'
        }

    async def _general_testing_task(self, task) -> Dict[str, Any]:
        """Handle general testing tasks"""
        return {
            'status': 'completed',
            'message': f'Testing task completed: {task.description}'
        }

    async def _setup_jest(self, project_path: Path):
        """Set up Jest testing framework"""
        # Jest is already configured in the other methods
        pass

    async def _setup_pytest(self, project_path: Path):
        """Set up Pytest for Python projects"""
        # Create pytest configuration
        pytest_ini = '''
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
'''
        with open(project_path / "pytest.ini", 'w') as f:
            f.write(pytest_ini)

    async def _setup_cypress(self, project_path: Path):
        """Set up Cypress for E2E testing"""
        # Cypress configuration is already created in _create_e2e_tests
        pass