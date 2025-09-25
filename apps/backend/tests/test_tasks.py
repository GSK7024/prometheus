#!/usr/bin/env python3
"""
Comprehensive test suite for the Task Management API.
Tests cover all endpoints, services, and edge cases using pytest and pytest-asyncio.
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient
from typing import AsyncGenerator, Generator

# Import application components
from apps.backend.app.main import app
from apps.backend.app.db import Base, get_db, Task
from apps.backend.app.services.task_service import TaskService
from apps.backend.app.models.task_models import (
    TaskCreate,
    TaskUpdate,
    TaskFilter,
    TaskStats
)

# Test database configuration
TEST_DATABASE_URL = "sqlite:///./test.db"

# Create test engine
test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

# Create test session factory
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    """Create test database tables."""
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def db_session() -> Generator:
    """Create database session for testing."""
    connection = test_engine.connect()
    transaction = connection.begin()
    session = TestSessionLocal(bind=connection)

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def task_service(db_session):
    """Create TaskService instance for testing."""
    return TaskService(db_session)


@pytest.fixture
def sample_task_data():
    """Sample task data for testing."""
    return {
        "title": "Test Task",
        "description": "This is a test task description",
        "completed": False
    }


@pytest.fixture
def sample_tasks():
    """Sample tasks for bulk testing."""
    return [
        {"title": "Task 1", "description": "Description 1", "completed": False},
        {"title": "Task 2", "description": "Description 2", "completed": True},
        {"title": "Task 3", "description": "Description 3", "completed": False},
    ]


# ===== SERVICE TESTS =====

class TestTaskService:
    """Test the TaskService business logic."""

    def test_create_task(self, task_service, sample_task_data):
        """Test creating a new task."""
        task_data = TaskCreate(**sample_task_data)
        result = task_service.create_task(task_data)

        assert result.title == sample_task_data["title"]
        assert result.description == sample_task_data["description"]
        assert result.completed == sample_task_data["completed"]
        assert result.id is not None

    def test_create_task_validation(self, task_service):
        """Test task creation validation."""
        # Empty title should raise ValueError
        with pytest.raises(ValueError, match="Title cannot be empty"):
            task_service.create_task(TaskCreate(title="", description="test"))

        # Long title should raise ValueError
        long_title = "a" * 256
        with pytest.raises(ValueError, match="cannot exceed 255 characters"):
            task_service.create_task(TaskCreate(title=long_title))

    def test_get_task(self, task_service, sample_task_data):
        """Test retrieving a task by ID."""
        # Create a task first
        task_data = TaskCreate(**sample_task_data)
        created = task_service.create_task(task_data)

        # Retrieve it
        retrieved = task_service.get_task(created.id)
        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.title == created.title

        # Test non-existent task
        assert task_service.get_task(999) is None

    def test_get_tasks(self, task_service, sample_tasks):
        """Test retrieving all tasks."""
        # Create multiple tasks
        for task_data in sample_tasks:
            task_service.create_task(TaskCreate(**task_data))

        # Get all tasks
        all_tasks = task_service.get_tasks()
        assert len(all_tasks) == len(sample_tasks)

        # Test filtering
        incomplete_tasks = task_service.get_tasks(TaskFilter(completed=False))
        assert len(incomplete_tasks) == 2  # Two incomplete tasks

        completed_tasks = task_service.get_tasks(TaskFilter(completed=True))
        assert len(completed_tasks) == 1  # One completed task

    def test_update_task(self, task_service, sample_task_data):
        """Test updating a task."""
        # Create a task
        created = task_service.create_task(TaskCreate(**sample_task_data))

        # Update it
        update_data = TaskUpdate(title="Updated Title", completed=True)
        updated = task_service.update_task(created.id, update_data)

        assert updated is not None
        assert updated.title == "Updated Title"
        assert updated.completed == True
        assert updated.id == created.id

        # Test non-existent task
        assert task_service.update_task(999, update_data) is None

    def test_delete_task(self, task_service, sample_task_data):
        """Test deleting a task."""
        # Create a task
        created = task_service.create_task(TaskCreate(**sample_task_data))

        # Delete it
        result = task_service.delete_task(created.id)
        assert result == True

        # Verify it's gone
        assert task_service.get_task(created.id) is None

        # Test deleting non-existent task
        assert task_service.delete_task(999) == False

    def test_mark_task_complete(self, task_service, sample_task_data):
        """Test marking task as complete."""
        # Create an incomplete task
        created = task_service.create_task(TaskCreate(**sample_task_data))
        assert created.completed == False

        # Mark as complete
        completed = task_service.mark_task_complete(created.id)
        assert completed is not None
        assert completed.completed == True

    def test_get_task_stats(self, task_service, sample_tasks):
        """Test getting task statistics."""
        # Create sample tasks
        for task_data in sample_tasks:
            task_service.create_task(TaskCreate(**task_data))

        stats = task_service.get_task_stats()
        assert stats.total == 3
        assert stats.completed == 1
        assert stats.incomplete == 2
        assert stats.completion_rate == (1/3) * 100

    def test_bulk_update_tasks(self, task_service, sample_tasks):
        """Test bulk updating tasks."""
        # Create tasks
        created_ids = []
        for task_data in sample_tasks:
            created = task_service.create_task(TaskCreate(**task_data))
            created_ids.append(created.id)

        # Bulk update to mark as complete
        update_data = TaskUpdate(completed=True)
        result = task_service.bulk_update_tasks(created_ids, update_data)

        assert result["total_processed"] == 3
        assert result["updated"] == 3
        assert result["not_found"] == 0

        # Verify all tasks are now complete
        for task_id in created_ids:
            task = task_service.get_task(task_id)
            assert task.completed == True


# ===== API TESTS =====

@pytest_asyncio.fixture
async def async_client() -> AsyncGenerator:
    """Create async test client."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def sync_client():
    """Create sync test client."""
    return TestClient(app)


class TestTaskAPI:
    """Test the REST API endpoints."""

    def test_health_check(self, sync_client):
        """Test health check endpoint."""
        response = sync_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_root_endpoint(self, sync_client):
        """Test root endpoint."""
        response = sync_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Task Management API"
        assert "docs" in data

    def test_create_task(self, sync_client, sample_task_data):
        """Test creating a task via API."""
        response = sync_client.post(
            "/api/v1/tasks",
            json=sample_task_data
        )
        assert response.status_code == 201
        data = response.json()
        assert data["title"] == sample_task_data["title"]
        assert data["description"] == sample_task_data["description"]
        assert data["completed"] == sample_task_data["completed"]
        assert "id" in data

    def test_create_task_validation(self, sync_client):
        """Test API validation for task creation."""
        # Empty title
        response = sync_client.post(
            "/api/v1/tasks",
            json={"title": "", "description": "test"}
        )
        assert response.status_code == 422

        # Long title
        response = sync_client.post(
            "/api/v1/tasks",
            json={"title": "a" * 256}
        )
        assert response.status_code == 422

    def test_get_tasks(self, sync_client):
        """Test getting all tasks via API."""
        # Create some tasks first
        tasks = [
            {"title": "Task 1", "description": "Desc 1", "completed": False},
            {"title": "Task 2", "description": "Desc 2", "completed": True},
        ]

        for task in tasks:
            sync_client.post("/api/v1/tasks", json=task)

        # Get all tasks
        response = sync_client.get("/api/v1/tasks")
        assert response.status_code == 200
        data = response.json()
        assert len(data["tasks"]) == 2
        assert data["total"] == 2

    def test_get_task_by_id(self, sync_client, sample_task_data):
        """Test getting a specific task by ID."""
        # Create a task
        create_response = sync_client.post("/api/v1/tasks", json=sample_task_data)
        task_id = create_response.json()["id"]

        # Get the task
        response = sync_client.get(f"/api/v1/tasks/{task_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == task_id
        assert data["title"] == sample_task_data["title"]

    def test_get_nonexistent_task(self, sync_client):
        """Test getting a non-existent task."""
        response = sync_client.get("/api/v1/tasks/999")
        assert response.status_code == 404

    def test_update_task(self, sync_client, sample_task_data):
        """Test updating a task via API."""
        # Create a task
        create_response = sync_client.post("/api/v1/tasks", json=sample_task_data)
        task_id = create_response.json()["id"]

        # Update it
        update_data = {"title": "Updated Title", "completed": True}
        response = sync_client.put(f"/api/v1/tasks/{task_id}", json=update_data)
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Updated Title"
        assert data["completed"] == True

    def test_delete_task(self, sync_client, sample_task_data):
        """Test deleting a task via API."""
        # Create a task
        create_response = sync_client.post("/api/v1/tasks", json=sample_task_data)
        task_id = create_response.json()["id"]

        # Delete it
        response = sync_client.delete(f"/api/v1/tasks/{task_id}")
        assert response.status_code == 204

        # Verify it's gone
        response = sync_client.get(f"/api/v1/tasks/{task_id}")
        assert response.status_code == 404

    def test_mark_task_complete(self, sync_client, sample_task_data):
        """Test marking task as complete via API."""
        # Create an incomplete task
        create_response = sync_client.post("/api/v1/tasks", json=sample_task_data)
        task_id = create_response.json()["id"]

        # Mark as complete
        response = sync_client.patch(f"/api/v1/tasks/{task_id}/complete")
        assert response.status_code == 200
        data = response.json()
        assert data["completed"] == True

    def test_get_task_stats(self, sync_client, sample_tasks):
        """Test getting task statistics via API."""
        # Create sample tasks
        for task in sample_tasks:
            sync_client.post("/api/v1/tasks", json=task)

        # Get stats
        response = sync_client.get("/api/v1/tasks/stats/summary")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert data["completed"] == 1
        assert data["incomplete"] == 2

    def test_filter_tasks(self, sync_client, sample_tasks):
        """Test filtering tasks via API."""
        # Create sample tasks
        for task in sample_tasks:
            sync_client.post("/api/v1/tasks", json=task)

        # Filter incomplete tasks
        response = sync_client.get("/api/v1/tasks?completed=false")
        assert response.status_code == 200
        data = response.json()
        assert len(data["tasks"]) == 2

        # Filter completed tasks
        response = sync_client.get("/api/v1/tasks?completed=true")
        assert response.status_code == 200
        data = response.json()
        assert len(data["tasks"]) == 1

    def test_pagination(self, sync_client):
        """Test pagination functionality."""
        # Create multiple tasks
        for i in range(5):
            sync_client.post("/api/v1/tasks", json={
                "title": f"Task {i}",
                "description": f"Description {i}",
                "completed": i % 2 == 0
            })

        # Get first page
        response = sync_client.get("/api/v1/tasks?page=1&page_size=2")
        assert response.status_code == 200
        data = response.json()
        assert len(data["tasks"]) == 2
        assert data["page"] == 1
        assert data["page_size"] == 2
        assert data["total"] == 5
        assert data["total_pages"] == 3

    def test_bulk_update(self, sync_client):
        """Test bulk update functionality."""
        # Create tasks
        task_ids = []
        for i in range(3):
            response = sync_client.post("/api/v1/tasks", json={
                "title": f"Task {i}",
                "description": f"Description {i}",
                "completed": False
            })
            task_ids.append(response.json()["id"])

        # Bulk update
        response = sync_client.put("/api/v1/tasks/bulk-update", params={
            "task_ids": task_ids,
            "completed": True
        })
        assert response.status_code == 200
        data = response.json()
        assert data["updated"] == 3
        assert data["not_found"] == 0
        assert data["total_processed"] == 3


# ===== ERROR HANDLING TESTS =====

class TestErrorHandling:
    """Test error handling for various edge cases."""

    def test_invalid_task_id(self, sync_client):
        """Test handling of invalid task IDs."""
        response = sync_client.get("/api/v1/tasks/invalid")
        assert response.status_code == 422

    def test_negative_task_id(self, sync_client):
        """Test handling of negative task IDs."""
        response = sync_client.get("/api/v1/tasks/-1")
        assert response.status_code == 422

    def test_empty_request_body(self, sync_client):
        """Test handling of empty request body."""
        response = sync_client.post("/api/v1/tasks", json={})
        assert response.status_code == 422

    def test_malformed_json(self, sync_client):
        """Test handling of malformed JSON."""
        response = sync_client.post(
            "/api/v1/tasks",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422