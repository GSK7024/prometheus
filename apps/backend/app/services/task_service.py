#!/usr/bin/env python3
"""
Task Service Module - Core business logic for task management.
This module contains all CRUD operations for tasks with proper validation and error handling.
"""

import logging
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from apps.backend.app.db import Task, get_db
from apps.backend.app.models.task_models import (
    TaskCreate,
    TaskUpdate,
    TaskResponse,
    TaskFilter,
    TaskStats
)

logger = logging.getLogger(__name__)


class TaskService:
    """Service class for managing task operations."""

    def __init__(self, db_session: Session):
        """Initialize task service with database session."""
        self.db = db_session

    def create_task(self, task_data: TaskCreate) -> TaskResponse:
        """Create a new task with the provided data."""
        try:
            # Validate input data
            if not task_data.title or not task_data.title.strip():
                raise ValueError("Task title cannot be empty")

            if len(task_data.title) > 255:
                raise ValueError("Task title cannot exceed 255 characters")

            if task_data.description and len(task_data.description) > 1000:
                raise ValueError("Task description cannot exceed 1000 characters")

            # Create new task instance
            db_task = Task(
                title=task_data.title.strip(),
                description=task_data.description.strip() if task_data.description else None,
                completed=task_data.completed
            )

            # Add to database
            self.db.add(db_task)
            self.db.commit()
            self.db.refresh(db_task)

            logger.info(f"Created new task with ID: {db_task.id}")
            return TaskResponse.from_orm(db_task)

        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error creating task: {e}")
            raise
        except Exception as e:
            logger.error(f"Error creating task: {e}")
            raise

    def get_task(self, task_id: int) -> Optional[TaskResponse]:
        """Retrieve a specific task by ID."""
        try:
            db_task = self.db.query(Task).filter(Task.id == task_id).first()
            if db_task:
                return TaskResponse.from_orm(db_task)
            return None
        except SQLAlchemyError as e:
            logger.error(f"Database error retrieving task {task_id}: {e}")
            raise

    def get_tasks(self, filters: Optional[TaskFilter] = None) -> List[TaskResponse]:
        """Retrieve all tasks with optional filtering."""
        try:
            query = self.db.query(Task)

            if filters:
                if filters.completed is not None:
                    query = query.filter(Task.completed == filters.completed)
                if filters.title_contains:
                    query = query.filter(Task.title.contains(filters.title_contains))

            # Order by creation date (newest first)
            query = query.order_by(Task.created_at.desc())

            db_tasks = query.all()
            return [TaskResponse.from_orm(task) for task in db_tasks]

        except SQLAlchemyError as e:
            logger.error(f"Database error retrieving tasks: {e}")
            raise

    def update_task(self, task_id: int, update_data: TaskUpdate) -> Optional[TaskResponse]:
        """Update an existing task."""
        try:
            db_task = self.db.query(Task).filter(Task.id == task_id).first()
            if not db_task:
                return None

            # Validate input data
            if update_data.title is not None:
                if not update_data.title.strip():
                    raise ValueError("Task title cannot be empty")
                if len(update_data.title) > 255:
                    raise ValueError("Task title cannot exceed 255 characters")
                db_task.title = update_data.title.strip()

            if update_data.description is not None:
                if len(update_data.description) > 1000:
                    raise ValueError("Task description cannot exceed 1000 characters")
                db_task.description = update_data.description.strip() if update_data.description else None

            if update_data.completed is not None:
                db_task.completed = update_data.completed

            self.db.commit()
            self.db.refresh(db_task)

            logger.info(f"Updated task with ID: {db_task.id}")
            return TaskResponse.from_orm(db_task)

        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error updating task {task_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error updating task {task_id}: {e}")
            raise

    def delete_task(self, task_id: int) -> bool:
        """Delete a task by ID."""
        try:
            db_task = self.db.query(Task).filter(Task.id == task_id).first()
            if not db_task:
                return False

            self.db.delete(db_task)
            self.db.commit()

            logger.info(f"Deleted task with ID: {task_id}")
            return True

        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error deleting task {task_id}: {e}")
            raise

    def mark_task_complete(self, task_id: int) -> Optional[TaskResponse]:
        """Mark a task as completed."""
        return self.update_task(task_id, TaskUpdate(completed=True))

    def mark_task_incomplete(self, task_id: int) -> Optional[TaskResponse]:
        """Mark a task as incomplete."""
        return self.update_task(task_id, TaskUpdate(completed=False))

    def get_task_stats(self) -> TaskStats:
        """Get statistics about tasks."""
        try:
            total_tasks = self.db.query(Task).count()
            completed_tasks = self.db.query(Task).filter(Task.completed == True).count()
            incomplete_tasks = total_tasks - completed_tasks

            return TaskStats(
                total=total_tasks,
                completed=completed_tasks,
                incomplete=incomplete_tasks,
                completion_rate=(completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
            )
        except SQLAlchemyError as e:
            logger.error(f"Database error getting task stats: {e}")
            raise

    def bulk_update_tasks(self, task_ids: List[int], update_data: TaskUpdate) -> Dict[str, Any]:
        """Update multiple tasks at once."""
        try:
            updated_count = 0
            not_found_count = 0

            for task_id in task_ids:
                result = self.update_task(task_id, update_data)
                if result:
                    updated_count += 1
                else:
                    not_found_count += 1

            return {
                "updated": updated_count,
                "not_found": not_found_count,
                "total_processed": len(task_ids)
            }
        except Exception as e:
            logger.error(f"Error in bulk update: {e}")
            raise