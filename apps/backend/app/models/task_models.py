#!/usr/bin/env python3
"""
Pydantic models for the Task Management API.
These models define the data structures for request/response handling.
"""

from typing import Optional, List
from pydantic import BaseModel, Field, validator
from datetime import datetime


class TaskCreate(BaseModel):
    """Model for creating a new task."""
    title: str = Field(..., min_length=1, max_length=255, description="Task title")
    description: Optional[str] = Field(None, max_length=1000, description="Task description")
    completed: bool = Field(False, description="Task completion status")

    @validator('title')
    def title_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Title cannot be empty')
        return v.strip()

    @validator('description')
    def description_must_not_be_empty(cls, v):
        if v is not None and not v.strip():
            return None
        return v.strip() if v else None

    class Config:
        schema_extra = {
            "example": {
                "title": "Complete project documentation",
                "description": "Write comprehensive documentation for the new API endpoints",
                "completed": False
            }
        }


class TaskUpdate(BaseModel):
    """Model for updating an existing task."""
    title: Optional[str] = Field(None, min_length=1, max_length=255, description="Updated task title")
    description: Optional[str] = Field(None, max_length=1000, description="Updated task description")
    completed: Optional[bool] = Field(None, description="Updated completion status")

    @validator('title')
    def title_must_not_be_empty(cls, v):
        if v is not None and not v.strip():
            raise ValueError('Title cannot be empty')
        return v.strip() if v else None

    @validator('description')
    def description_must_not_be_empty(cls, v):
        if v is not None and not v.strip():
            return None
        return v.strip() if v else None

    class Config:
        schema_extra = {
            "example": {
                "title": "Complete project documentation",
                "description": "Write comprehensive documentation for the new API endpoints",
                "completed": True
            }
        }


class TaskResponse(BaseModel):
    """Model for task response data."""
    id: int
    title: str
    description: Optional[str]
    completed: bool
    created_at: Optional[datetime]
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TaskFilter(BaseModel):
    """Model for filtering tasks."""
    completed: Optional[bool] = Field(None, description="Filter by completion status")
    title_contains: Optional[str] = Field(None, description="Filter by title containing this text")

    class Config:
        schema_extra = {
            "example": {
                "completed": False,
                "title_contains": "documentation"
            }
        }


class TaskStats(BaseModel):
    """Model for task statistics."""
    total: int
    completed: int
    incomplete: int
    completion_rate: float

    class Config:
        schema_extra = {
            "example": {
                "total": 25,
                "completed": 10,
                "incomplete": 15,
                "completion_rate": 40.0
            }
        }


class TaskListResponse(BaseModel):
    """Model for paginated task list response."""
    tasks: List[TaskResponse]
    total: int
    page: int
    page_size: int
    total_pages: int

    @classmethod
    def from_tasks(cls, tasks: List[TaskResponse], total: int, page: int, page_size: int) -> "TaskListResponse":
        return cls(
            tasks=tasks,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=(total + page_size - 1) // page_size
        )


class ErrorResponse(BaseModel):
    """Model for error responses."""
    error: str
    message: Optional[str] = None
    details: Optional[dict] = None

    class Config:
        schema_extra = {
            "example": {
                "error": "TaskNotFoundError",
                "message": "Task with ID 123 not found",
                "details": {"task_id": 123}
            }
        }


class BulkUpdateResponse(BaseModel):
    """Model for bulk update operation response."""
    updated: int
    not_found: int
    total_processed: int

    class Config:
        schema_extra = {
            "example": {
                "updated": 3,
                "not_found": 1,
                "total_processed": 4
            }
        }