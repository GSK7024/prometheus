# Task Management API - FastAPI Backend

A modern, RESTful API for managing tasks built with FastAPI, SQLAlchemy, and PostgreSQL. This is part of the Project Phoenix transformation from the legacy Flask application to a unified, scalable backend architecture.

## 🚀 Features

- **RESTful API**: Complete CRUD operations for task management
- **Modern Framework**: Built with FastAPI for high performance and automatic documentation
- **Database Integration**: Persistent storage with PostgreSQL and SQLAlchemy ORM
- **Comprehensive Validation**: Pydantic models with robust input validation
- **Error Handling**: Structured error responses with detailed information
- **Pagination Support**: Efficient handling of large task lists
- **CORS Support**: Ready for frontend integration
- **Health Monitoring**: Built-in health check endpoints
- **Comprehensive Logging**: Structured logging for debugging and monitoring

## 📋 API Endpoints

### Tasks

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/tasks` | Get all tasks with optional filtering |
| GET | `/api/v1/tasks/{task_id}` | Get a specific task by ID |
| POST | `/api/v1/tasks` | Create a new task |
| PUT | `/api/v1/tasks/{task_id}` | Update an existing task |
| DELETE | `/api/v1/tasks/{task_id}` | Delete a task |
| PATCH | `/api/v1/tasks/{task_id}/complete` | Mark task as complete |
| PATCH | `/api/v1/tasks/{task_id}/incomplete` | Mark task as incomplete |
| GET | `/api/v1/tasks/stats/summary` | Get task statistics |
| PUT | `/api/v1/tasks/bulk-update` | Update multiple tasks |

### Health & Info

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information and documentation links |
| GET | `/health` | Health check endpoint |

## 🏗️ Architecture

```
apps/backend/
├── app/
│   ├── main.py                 # FastAPI application entry point
│   ├── db.py                   # Database configuration and models
│   ├── models/
│   │   └── task_models.py      # Pydantic models for request/response
│   ├── services/
│   │   └── task_service.py     # Business logic layer
│   └── routers/
│       └── tasks.py            # REST API endpoints
├── tests/
│   └── test_tasks.py           # Comprehensive test suite
└── requirements.txt            # Python dependencies
```

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.11+
- PostgreSQL database
- Virtual environment (recommended)

### 1. Clone and Navigate

```bash
cd apps/backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Create a `.env` file:

```bash
DATABASE_URL=postgresql://username:password@localhost/taskdb
DEBUG=true
PORT=8000
```

### 4. Setup Database

```bash
# Create PostgreSQL database
createdb taskdb

# Run database migrations
python -c "from app.db import create_tables; create_tables()"
```

### 5. Run the Application

**Development mode:**
```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Production mode:**
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 🧪 Testing

### Run Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=app

# Specific test file
pytest tests/test_tasks.py

# Verbose output
pytest -v
```

### Test Structure

- **Unit Tests**: Test individual service methods
- **Integration Tests**: Test API endpoints with database
- **Mock Tests**: Test error handling and edge cases

## 📚 API Usage Examples

### Create a Task

```bash
curl -X POST "http://localhost:8000/api/v1/tasks" \
     -H "Content-Type: application/json" \
     -d '{
       "title": "Complete project documentation",
       "description": "Write comprehensive docs for the new API",
       "completed": false
     }'
```

### Get All Tasks

```bash
curl -X GET "http://localhost:8000/api/v1/tasks"
```

### Get Tasks with Filters

```bash
# Get only incomplete tasks
curl -X GET "http://localhost:8000/api/v1/tasks?completed=false"

# Get tasks containing "doc" in title
curl -X GET "http://localhost:8000/api/v1/tasks?title_contains=doc"
```

### Update a Task

```bash
curl -X PUT "http://localhost:8000/api/v1/tasks/1" \
     -H "Content-Type: application/json" \
     -d '{
       "title": "Updated task title",
       "completed": true
     }'
```

### Mark Task Complete

```bash
curl -X PATCH "http://localhost:8000/api/v1/tasks/1/complete"
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql://localhost/taskdb` | PostgreSQL connection string |
| `DEBUG` | `false` | Enable debug mode |
| `PORT` | `8000` | Server port |
| `HOST` | `0.0.0.0` | Server host |
| `SQLALCHEMY_ECHO` | `false` | Enable SQL query logging |

### Database Configuration

The application uses SQLAlchemy with the following features:
- Connection pooling (pool_size=10, max_overflow=20)
- Connection validation (pool_pre_ping=True)
- Automatic table creation on startup
- Comprehensive error handling

## 🚀 Deployment

### Docker (Recommended)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ ./app/
COPY .env ./

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0"]
```

### Production Considerations

1. **Security**: Configure proper CORS origins, enable HTTPS
2. **Performance**: Configure connection pooling, enable caching
3. **Monitoring**: Add health checks, logging, and metrics
4. **Scalability**: Use load balancer, database read replicas

## 📊 Monitoring & Logging

- **Structured Logging**: Uses Python's logging module with structured format
- **Health Checks**: Built-in `/health` endpoint
- **Error Tracking**: Comprehensive error handling with detailed responses
- **Performance Monitoring**: Request/response logging

## 🤝 Contributing

1. Follow the existing code style and structure
2. Add comprehensive tests for new features
3. Update documentation as needed
4. Ensure all tests pass before submitting PR

## 📄 License

This project is part of the Project Phoenix transformation initiative.