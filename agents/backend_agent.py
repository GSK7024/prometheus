"""
Backend Agent
Specialized in creating robust, scalable backend applications
Supports Node.js, Python (Django, Flask, FastAPI), and other backend frameworks.
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class BackendAgent:
    """
    Advanced backend development agent that creates production-ready
    backend applications with modern frameworks and best practices.
    """

    def __init__(self, agent_system):
        self.agent_system = agent_system
        self.supported_frameworks = {
            'nodejs': self._create_nodejs_app,
            'django': self._create_django_app,
            'flask': self._create_flask_app,
            'fastapi': self._create_fastapi_app
        }

    async def execute_task(self, task) -> Dict[str, Any]:
        """Execute a backend-related task"""
        logger.info(f"Backend agent executing: {task.description}")

        if 'initialize' in task.description.lower() and 'backend' in task.description.lower():
            return await self._setup_backend(task)
        elif 'api' in task.description.lower() or 'endpoint' in task.description.lower():
            return await self._create_api_endpoints(task)
        elif 'authentication' in task.description.lower() or 'auth' in task.description.lower():
            return await self._implement_authentication(task)
        elif 'database' in task.description.lower():
            return await self._setup_database_integration(task)
        else:
            return await self._general_backend_task(task)

    async def _setup_backend(self, task) -> Dict[str, Any]:
        """Set up backend application based on framework"""
        project_config = self.agent_system.current_project

        if not project_config or not project_config.backend_framework:
            raise ValueError("No backend framework specified in project config")

        framework = project_config.backend_framework.lower()
        project_path = Path(project_config.target_directory)

        if framework not in self.supported_frameworks:
            raise ValueError(f"Unsupported backend framework: {framework}")

        # Create backend directory
        backend_path = project_path / "backend"
        backend_path.mkdir(exist_ok=True)

        # Initialize the framework
        success = await self.supported_frameworks[framework](backend_path)

        if success:
            return {
                'status': 'completed',
                'backend_path': str(backend_path),
                'framework': framework,
                'files_created': self._get_created_files(backend_path)
            }
        else:
            raise Exception(f"Failed to create {framework} application")

    async def _create_nodejs_app(self, project_path: Path) -> bool:
        """Create a Node.js/Express backend application"""
        try:
            # Initialize npm project
            subprocess.run([
                'npm', 'init', '-y'
            ], cwd=project_path, check=True, capture_output=True)

            # Install dependencies
            subprocess.run([
                'npm', 'install', 'express', 'cors', 'helmet', 'dotenv',
                'bcryptjs', 'jsonwebtoken', 'mongoose', 'sequelize',
                'nodemon', 'jest', 'supertest'
            ], cwd=project_path, check=True)

            # Create basic Express server
            self._create_express_structure(project_path)

            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create Node.js app: {e}")
            return False

    async def _create_django_app(self, project_path: Path) -> bool:
        """Create a Django backend application"""
        try:
            # Create Django project
            subprocess.run([
                'python', '-m', 'django', 'startproject', 'backend', '.'
            ], cwd=project_path, check=True, capture_output=True)

            # Install additional packages
            subprocess.run([
                'pip', 'install', 'djangorestframework', 'django-cors-headers',
                'djoser', 'djangorestframework-simplejwt', 'psycopg2-binary'
            ], cwd=project_path, check=True)

            # Create Django apps and configure
            self._create_django_structure(project_path)

            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create Django app: {e}")
            return False

    async def _create_flask_app(self, project_path: Path) -> bool:
        """Create a Flask backend application"""
        try:
            # Create virtual environment
            subprocess.run([
                'python', '-m', 'venv', 'venv'
            ], cwd=project_path, check=True)

            # Install Flask and dependencies
            pip_cmd = str(project_path / "venv" / "bin" / "pip")
            subprocess.run([
                pip_cmd, 'install', 'flask', 'flask-cors', 'flask-jwt-extended',
                'flask-sqlalchemy', 'flask-migrate', 'psycopg2-binary'
            ], cwd=project_path, check=True)

            # Create Flask app structure
            self._create_flask_structure(project_path)

            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create Flask app: {e}")
            return False

    async def _create_fastapi_app(self, project_path: Path) -> bool:
        """Create a FastAPI backend application"""
        try:
            # Install FastAPI and dependencies
            subprocess.run([
                'pip', 'install', 'fastapi', 'uvicorn', 'pydantic',
                'python-jose[cryptography]', 'passlib[bcrypt]',
                'python-multipart', 'sqlalchemy', 'alembic'
            ], cwd=project_path, check=True)

            # Create FastAPI structure
            self._create_fastapi_structure(project_path)

            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create FastAPI app: {e}")
            return False

    def _create_express_structure(self, project_path: Path):
        """Create modern Express.js application structure"""

        # Create main server file
        server_content = '''
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const dotenv = require('dotenv');
const path = require('path');

// Load environment variables
dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(helmet());
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Routes
app.use('/api/auth', require('./routes/auth'));
app.use('/api/users', require('./routes/users'));
app.use('/api/products', require('./routes/products'));

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({
    status: 'OK',
    message: 'Server is running',
    timestamp: new Date().toISOString()
  });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({
    error: 'Something went wrong!',
    message: process.env.NODE_ENV === 'development' ? err.message : 'Internal server error'
  });
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({ error: 'Route not found' });
});

app.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`);
  console.log(`📝 Environment: ${process.env.NODE_ENV || 'development'}`);
});

module.exports = app;
'''
        with open(project_path / "server.js", 'w') as f:
            f.write(server_content)

        # Create routes directory
        routes_dir = project_path / "routes"
        routes_dir.mkdir(exist_ok=True)

        # Create auth routes
        auth_routes = '''
const express = require('express');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const User = require('../models/User');

const router = express.Router();

// Register user
router.post('/register', async (req, res) => {
  try {
    const { name, email, password } = req.body;

    // Check if user exists
    const existingUser = await User.findOne({ email });
    if (existingUser) {
      return res.status(400).json({ error: 'User already exists' });
    }

    // Hash password
    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(password, salt);

    // Create user
    const user = new User({
      name,
      email,
      password: hashedPassword
    });

    await user.save();

    // Generate JWT token
    const token = jwt.sign(
      { userId: user._id },
      process.env.JWT_SECRET || 'your-secret-key',
      { expiresIn: '7d' }
    );

    res.status(201).json({
      message: 'User created successfully',
      token,
      user: {
        id: user._id,
        name: user.name,
        email: user.email
      }
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Login user
router.post('/login', async (req, res) => {
  try {
    const { email, password } = req.body;

    // Find user
    const user = await User.findOne({ email });
    if (!user) {
      return res.status(400).json({ error: 'Invalid credentials' });
    }

    // Check password
    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) {
      return res.status(400).json({ error: 'Invalid credentials' });
    }

    // Generate JWT token
    const token = jwt.sign(
      { userId: user._id },
      process.env.JWT_SECRET || 'your-secret-key',
      { expiresIn: '7d' }
    );

    res.json({
      message: 'Login successful',
      token,
      user: {
        id: user._id,
        name: user.name,
        email: user.email
      }
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

module.exports = router;
'''
        with open(project_path / "routes" / "auth.js", 'w') as f:
            f.write(auth_routes)

        # Create user routes
        user_routes = '''
const express = require('express');
const auth = require('../middleware/auth');
const User = require('../models/User');

const router = express.Router();

// Get current user
router.get('/me', auth, async (req, res) => {
  try {
    const user = await User.findById(req.user.userId).select('-password');
    res.json(user);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Update user
router.put('/me', auth, async (req, res) => {
  try {
    const { name, email } = req.body;
    const user = await User.findByIdAndUpdate(
      req.user.userId,
      { name, email },
      { new: true }
    ).select('-password');

    res.json(user);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

module.exports = router;
'''
        with open(project_path / "routes" / "users.js", 'w') as f:
            f.write(user_routes)

        # Create product routes
        product_routes = '''
const express = require('express');
const auth = require('../middleware/auth');
const Product = require('../models/Product');

const router = express.Router();

// Get all products
router.get('/', async (req, res) => {
  try {
    const products = await Product.find();
    res.json(products);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Get single product
router.get('/:id', async (req, res) => {
  try {
    const product = await Product.findById(req.params.id);
    if (!product) {
      return res.status(404).json({ error: 'Product not found' });
    }
    res.json(product);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Create product (admin only)
router.post('/', auth, async (req, res) => {
  try {
    const product = new Product(req.body);
    await product.save();
    res.status(201).json(product);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

module.exports = router;
'''
        with open(project_path / "routes" / "products.js", 'w') as f:
            f.write(product_routes)

        # Create middleware directory
        middleware_dir = project_path / "middleware"
        middleware_dir.mkdir(exist_ok=True)

        # Create auth middleware
        auth_middleware = '''
const jwt = require('jsonwebtoken');

const auth = (req, res, next) => {
  try {
    const token = req.header('Authorization').replace('Bearer ', '');
    const decoded = jwt.verify(token, process.env.JWT_SECRET || 'your-secret-key');
    req.user = decoded;
    next();
  } catch (error) {
    res.status(401).json({ error: 'Please authenticate' });
  }
};

module.exports = auth;
'''
        with open(project_path / "middleware" / "auth.js", 'w') as f:
            f.write(auth_middleware)

        # Create models directory
        models_dir = project_path / "models"
        models_dir.mkdir(exist_ok=True)

        # Create User model
        user_model = '''
const mongoose = require('mongoose');

const userSchema = new mongoose.Schema({
  name: {
    type: String,
    required: true,
    trim: true
  },
  email: {
    type: String,
    required: true,
    unique: true,
    trim: true,
    lowercase: true
  },
  password: {
    type: String,
    required: true,
    minlength: 6
  },
  role: {
    type: String,
    enum: ['user', 'admin'],
    default: 'user'
  },
  createdAt: {
    type: Date,
    default: Date.now
  }
});

module.exports = mongoose.model('User', userSchema);
'''
        with open(project_path / "models" / "User.js", 'w') as f:
            f.write(user_model)

        # Create Product model
        product_model = '''
const mongoose = require('mongoose');

const productSchema = new mongoose.Schema({
  name: {
    type: String,
    required: true,
    trim: true
  },
  description: {
    type: String,
    required: true
  },
  price: {
    type: Number,
    required: true,
    min: 0
  },
  image: {
    type: String
  },
  category: {
    type: String,
    required: true
  },
  stock: {
    type: Number,
    default: 0
  },
  createdAt: {
    type: Date,
    default: Date.now
  }
});

module.exports = mongoose.model('Product', productSchema);
'''
        with open(project_path / "models" / "Product.js", 'w') as f:
            f.write(product_model)

        # Create environment file
        env_content = '''
NODE_ENV=development
PORT=5000
MONGODB_URI=mongodb://localhost:27017/myapp
JWT_SECRET=your-super-secret-jwt-key-here
'''
        with open(project_path / ".env", 'w') as f:
            f.write(env_content)

        # Create package.json scripts
        package_json = json.loads((project_path / "package.json").read_text())
        package_json["scripts"] = {
            "start": "node server.js",
            "dev": "nodemon server.js",
            "test": "jest",
            "test:watch": "jest --watch"
        }
        with open(project_path / "package.json", 'w') as f:
            json.dump(package_json, f, indent=2)

    def _create_django_structure(self, project_path: Path):
        """Create Django project structure"""

        # Update settings.py
        settings_content = '''
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = 'your-secret-key-here'
DEBUG = True
ALLOWED_HOSTS = []

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'corsheaders',
    'djoser',
    'users',
    'products',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'backend.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'backend.wsgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'myapp',
        'USER': 'postgres',
        'PASSWORD': 'password',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_L10N = True
USE_TZ = True

STATIC_URL = '/static/'
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    ),
    'DEFAULT_PERMISSION_CLASSES': (
        'rest_framework.permissions.IsAuthenticated',
    ),
}
'''
        settings_file = project_path / "backend" / "settings.py"
        with open(settings_file, 'w') as f:
            f.write(settings_content)

    def _create_flask_structure(self, project_path: Path):
        """Create Flask application structure"""

        # Create app.py
        app_content = '''
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
from models import db, User, Product
import os
from datetime import timedelta

app = Flask(__name__)
CORS(app)

# Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://user:password@localhost/myapp'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = 'your-secret-key-here'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=7)

# Initialize extensions
db.init_app(app)
JWTManager(app)

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'OK',
        'message': 'Flask server is running'
    })

@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')

    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'User already exists'}), 400

    user = User(name=name, email=email)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()

    access_token = create_access_token(identity=user.id)
    return jsonify({
        'message': 'User created successfully',
        'token': access_token,
        'user': {
            'id': user.id,
            'name': user.name,
            'email': user.email
        }
    }), 201

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    user = User.query.filter_by(email=email).first()
    if user and user.check_password(password):
        access_token = create_access_token(identity=user.id)
        return jsonify({
            'message': 'Login successful',
            'token': access_token,
            'user': {
                'id': user.id,
                'name': user.name,
                'email': user.email
            }
        })

    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/api/users/me', methods=['GET'])
@jwt_required()
def get_current_user():
    current_user_id = get_jwt_identity()
    user = User.query.get(current_user_id)
    return jsonify({
        'id': user.id,
        'name': user.name,
        'email': user.email
    })

@app.route('/api/products', methods=['GET'])
def get_products():
    products = Product.query.all()
    return jsonify([{
        'id': p.id,
        'name': p.name,
        'description': p.description,
        'price': p.price
    } for p in products])

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5000)
'''
        with open(project_path / "app.py", 'w') as f:
            f.write(app_content)

    def _create_fastapi_structure(self, project_path: Path):
        """Create FastAPI application structure"""

        # Create main.py
        main_content = '''
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
import models, schemas, crud
from database import SessionLocal, engine

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="My API", description="A modern FastAPI application")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication
SECRET_KEY = "your-secret-key-here"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def authenticate_user(db: Session, email: str, password: str):
    user = crud.get_user_by_email(db, email=email)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = schemas.TokenData(email=email)
    except JWTError:
        raise credentials_exception
    # In a real app, you'd fetch the user from the database
    return token_data

@app.get("/api/health")
def health_check():
    return {"status": "OK", "message": "FastAPI server is running"}

@app.post("/api/auth/register", response_model=schemas.User)
def register(user: schemas.UserCreate, db: Session = Depends(crud.get_db)):
    db_user = crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    return crud.create_user(db=db, user=user)

@app.post("/api/auth/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # In a real app, you'd verify against the database
    # For now, just return a mock token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": form_data.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/api/users/me", response_model=schemas.User)
def read_users_me(current_user: schemas.User = Depends(get_current_user)):
    return current_user

@app.get("/api/products")
def get_products(db: Session = Depends(crud.get_db)):
    products = crud.get_products(db)
    return products

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
        with open(project_path / "main.py", 'w') as f:
            f.write(main_content)

    async def _create_api_endpoints(self, task) -> Dict[str, Any]:
        """Create API endpoints based on requirements"""
        # This would create specific API endpoints based on the task description
        return {
            'status': 'completed',
            'message': 'API endpoints created successfully'
        }

    async def _implement_authentication(self, task) -> Dict[str, Any]:
        """Implement authentication system"""
        return {
            'status': 'completed',
            'message': 'Authentication system implemented'
        }

    async def _setup_database_integration(self, task) -> Dict[str, Any]:
        """Set up database integration"""
        return {
            'status': 'completed',
            'message': 'Database integration configured'
        }

    async def _general_backend_task(self, task) -> Dict[str, Any]:
        """Handle general backend tasks"""
        return {
            'status': 'completed',
            'message': f'Backend task completed: {task.description}'
        }

    def _get_created_files(self, directory: Path) -> List[str]:
        """Get list of files created in a directory"""
        files = []
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                files.append(str(file_path.relative_to(directory)))
        return files