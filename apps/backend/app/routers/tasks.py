#!/usr/bin/env python3
"""
Task Router - RESTful API endpoints for task management.
This module defines all HTTP endpoints for CRUD operations on tasks.
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from apps.backend.app.db import get_db
from apps.backend.app.services.task_service import TaskService
from apps.backend.app.models.task_models import (
    TaskCreate,
    TaskUpdate,
    TaskResponse,
    TaskFilter,
    TaskStats,
    TaskListResponse,
    ErrorResponse,
    BulkUpdateResponse
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/api/v1/tasks",
    tags=["tasks"],
    responses={
        404: {"model": ErrorResponse, "description": "Task not found"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)


def get_task_service(db: Session = Depends(get_db)) -> TaskService:
    """Dependency to get task service instance."""
    return TaskService(db)


@router.get("/", response_model=TaskListResponse, summary="Get all tasks")
async def get_tasks(
    completed: Optional[bool] = Query(None, description="Filter by completion status"),
    title_contains: Optional[str] = Query(None, description="Filter by title containing text"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page"),
    task_service: TaskService = Depends(get_task_service)
) -> TaskListResponse:
    """
    Retrieve all tasks with optional filtering and pagination.

    - **completed**: Filter by completion status (true/false)
    - **title_contains**: Filter tasks containing this text in title
    - **page**: Page number (starts from 1)
    - **page_size**: Number of items per page (max 100)
    """
    try:
        filters = TaskFilter(
            completed=completed,
            title_contains=title_contains
        )

        all_tasks = task_service.get_tasks(filters)
        total = len(all_tasks)

        # Calculate pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_tasks = all_tasks[start_idx:end_idx]

        return TaskListResponse.from_tasks(paginated_tasks, total, page, page_size)

    except Exception as e:
        logger.error(f"Error retrieving tasks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving tasks"
        )


@router.get("/{task_id}", response_model=TaskResponse, summary="Get task by ID")
async def get_task(
    task_id: int,
    task_service: TaskService = Depends(get_task_service)
) -> TaskResponse:
    """
    Retrieve a specific task by its ID.

    - **task_id**: Unique identifier of the task
    """
    try:
        task = task_service.get_task(task_id)
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task with ID {task_id} not found"
            )
        return task
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving task"
        )


@router.post("/", response_model=TaskResponse, status_code=status.HTTP_201_CREATED, summary="Create new task")
async def create_task(
    task_data: TaskCreate,
    task_service: TaskService = Depends(get_task_service)
) -> TaskResponse:
    """
    Create a new task with the provided data.

    - **title**: Required task title (max 255 characters)
    - **description**: Optional task description (max 1000 characters)
    - **completed**: Task completion status (default: false)
    """
    try:
        return task_service.create_task(task_data)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error creating task"
        )


@router.put("/{task_id}", response_model=TaskResponse, summary="Update task")
async def update_task(
    task_id: int,
    task_data: TaskUpdate,
    task_service: TaskService = Depends(get_task_service)
) -> TaskResponse:
    """
    Update an existing task.

    - **task_id**: Unique identifier of the task to update
    - **title**: Optional updated task title
    - **description**: Optional updated task description
    - **completed**: Optional updated completion status
    """
    try:
        task = task_service.update_task(task_id, task_data)
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task with ID {task_id} not found"
            )
        return task
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error updating task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error updating task"
        )


@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete task")
async def delete_task(
    task_id: int,
    task_service: TaskService = Depends(get_task_service)
):
    """
    Delete a task by its ID.

    - **task_id**: Unique identifier of the task to delete
    """
    try:
        success = task_service.delete_task(task_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task with ID {task_id} not found"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error deleting task"
        )


@router.patch("/{task_id}/complete", response_model=TaskResponse, summary="Mark task as complete")
async def mark_task_complete(
    task_id: int,
    task_service: TaskService = Depends(get_task_service)
) -> TaskResponse:
    """
    Mark a specific task as completed.

    - **task_id**: Unique identifier of the task to mark as complete
    """
    try:
        task = task_service.mark_task_complete(task_id)
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task with ID {task_id} not found"
            )
        return task
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error marking task {task_id} as complete: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error updating task"
        )


@router.patch("/{task_id}/incomplete", response_model=TaskResponse, summary="Mark task as incomplete")
async def mark_task_incomplete(
    task_id: int,
    task_service: TaskService = Depends(get_task_service)
) -> TaskResponse:
    """
    Mark a specific task as incomplete.

    - **task_id**: Unique identifier of the task to mark as incomplete
    """
    try:
        task = task_service.mark_task_incomplete(task_id)
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task with ID {task_id} not found"
            )
        return task
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error marking task {task_id} as incomplete: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error updating task"
        )


@router.get("/stats/summary", response_model=TaskStats, summary="Get task statistics")
async def get_task_stats(
    task_service: TaskService = Depends(get_task_service)
) -> TaskStats:
    """
    Get comprehensive statistics about all tasks.

    Returns:
    - Total number of tasks
    - Number of completed tasks
    - Number of incomplete tasks
    - Completion rate percentage
    """
    try:
        return task_service.get_task_stats()
    except Exception as e:
        logger.error(f"Error retrieving task stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving task statistics"
        )


@router.put("/bulk-update", response_model=BulkUpdateResponse, summary="Bulk update tasks")
async def bulk_update_tasks(
    task_ids: List[int] = Query(..., description="List of task IDs to update"),
    completed: Optional[bool] = Query(None, description="New completion status"),
    task_service: TaskService = Depends(get_task_service)
) -> BulkUpdateResponse:
    """
    Update multiple tasks at once.

    - **task_ids**: List of task IDs to update
    - **completed**: Optional new completion status for all tasks
    """
    try:
        update_data = TaskUpdate(completed=completed)
        result = task_service.bulk_update_tasks(task_ids, update_data)
        return BulkUpdateResponse(**result)
    except Exception as e:
        logger.error(f"Error in bulk update: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error in bulk update operation"
        )