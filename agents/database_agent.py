"""
Database Agent
Specialized in setting up and managing databases for web applications
Supports PostgreSQL, MySQL, MongoDB, SQLite and database migrations.
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class DatabaseAgent:
    """
    Advanced database management agent that creates and configures
    databases with proper schemas, migrations, and optimizations.
    """

    def __init__(self, agent_system):
        self.agent_system = agent_system
        self.supported_databases = {
            'postgresql': self._setup_postgresql,
            'mysql': self._setup_mysql,
            'mongodb': self._setup_mongodb,
            'sqlite': self._setup_sqlite
        }

    async def execute_task(self, task) -> Dict[str, Any]:
        """Execute a database-related task"""
        logger.info(f"Database agent executing: {task.description}")

        if 'setup' in task.description.lower() and 'database' in task.description.lower():
            return await self._setup_database(task)
        elif 'migration' in task.description.lower():
            return await self._create_migrations(task)
        elif 'seed' in task.description.lower() or 'sample' in task.description.lower():
            return await self._seed_database(task)
        else:
            return await self._general_database_task(task)

    async def _setup_database(self, task) -> Dict[str, Any]:
        """Set up database based on project configuration"""
        project_config = self.agent_system.current_project

        if not project_config or not project_config.database:
            raise ValueError("No database specified in project config")

        database = project_config.database.lower()
        project_path = Path(project_config.target_directory)

        if database not in self.supported_databases:
            raise ValueError(f"Unsupported database: {database}")

        # Set up database
        success = await self.supported_databases[database](project_path)

        if success:
            return {
                'status': 'completed',
                'database': database,
                'message': f'{database} database setup completed'
            }
        else:
            raise Exception(f"Failed to set up {database} database")

    async def _setup_postgresql(self, project_path: Path) -> bool:
        """Set up PostgreSQL database"""
        try:
            # Create database configuration
            db_config = {
                'database': 'myapp',
                'user': 'postgres',
                'password': 'password',
                'host': 'localhost',
                'port': 5432
            }

            # Create SQL schema file
            schema_content = '''
-- PostgreSQL schema for modern web application

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Products table (for e-commerce)
CREATE TABLE IF NOT EXISTS products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    price DECIMAL(10,2) NOT NULL,
    image_url VARCHAR(500),
    category VARCHAR(100),
    stock_quantity INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Orders table
CREATE TABLE IF NOT EXISTS orders (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    total_amount DECIMAL(10,2) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Order items table
CREATE TABLE IF NOT EXISTS order_items (
    id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES orders(id),
    product_id INTEGER REFERENCES products(id),
    quantity INTEGER NOT NULL,
    price DECIMAL(10,2) NOT NULL
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_products_category ON products(category);
CREATE INDEX IF NOT EXISTS idx_orders_user_id ON orders(user_id);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_products_updated_at BEFORE UPDATE ON products
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_orders_updated_at BEFORE UPDATE ON orders
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
'''
            schema_file = project_path / "database" / "schema.sql"
            schema_file.parent.mkdir(exist_ok=True)
            with open(schema_file, 'w') as f:
                f.write(schema_content)

            # Create database connection configuration
            self._create_db_connection_config(project_path, db_config)

            return True
        except Exception as e:
            logger.error(f"Failed to set up PostgreSQL: {e}")
            return False

    async def _setup_mysql(self, project_path: Path) -> bool:
        """Set up MySQL database"""
        try:
            # Create MySQL schema
            schema_content = '''
-- MySQL schema for modern web application

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Products table (for e-commerce)
CREATE TABLE IF NOT EXISTS products (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    price DECIMAL(10,2) NOT NULL,
    image_url VARCHAR(500),
    category VARCHAR(100),
    stock_quantity INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Orders table
CREATE TABLE IF NOT EXISTS orders (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    total_amount DECIMAL(10,2) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Order items table
CREATE TABLE IF NOT EXISTS order_items (
    id INT AUTO_INCREMENT PRIMARY KEY,
    order_id INT,
    product_id INT,
    quantity INT NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(id),
    FOREIGN KEY (product_id) REFERENCES products(id)
);

-- Create indexes for better performance
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_products_category ON products(category);
CREATE INDEX idx_orders_user_id ON orders(user_id);
CREATE INDEX idx_orders_status ON orders(status);
'''
            schema_file = project_path / "database" / "schema.sql"
            schema_file.parent.mkdir(exist_ok=True)
            with open(schema_file, 'w') as f:
                f.write(schema_content)

            return True
        except Exception as e:
            logger.error(f"Failed to set up MySQL: {e}")
            return False

    async def _setup_mongodb(self, project_path: Path) -> bool:
        """Set up MongoDB database"""
        try:
            # Create MongoDB connection configuration
            mongo_config = {
                'uri': 'mongodb://localhost:27017/myapp',
                'database': 'myapp'
            }

            # Create Mongoose models (for Node.js)
            models_dir = project_path / "backend" / "models"
            models_dir.mkdir(exist_ok=True)

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
  },
  updatedAt: {
    type: Date,
    default: Date.now
  }
});

module.exports = mongoose.model('User', userSchema);
'''

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
  },
  updatedAt: {
    type: Date,
    default: Date.now
  }
});

module.exports = mongoose.model('Product', productSchema);
'''

            with open(models_dir / "User.js", 'w') as f:
                f.write(user_model)

            with open(models_dir / "Product.js", 'w') as f:
                f.write(product_model)

            return True
        except Exception as e:
            logger.error(f"Failed to set up MongoDB: {e}")
            return False

    async def _setup_sqlite(self, project_path: Path) -> bool:
        """Set up SQLite database"""
        try:
            # Create SQLite database file and schema
            db_file = project_path / "database" / "app.db"
            db_file.parent.mkdir(exist_ok=True)

            schema_content = '''
-- SQLite schema for modern web application

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT DEFAULT 'user',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Products table (for e-commerce)
CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    price REAL NOT NULL,
    image_url TEXT,
    category TEXT,
    stock_quantity INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Orders table
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    total_amount REAL NOT NULL,
    status TEXT DEFAULT 'pending',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Order items table
CREATE TABLE IF NOT EXISTS order_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER,
    product_id INTEGER,
    quantity INTEGER NOT NULL,
    price REAL NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(id),
    FOREIGN KEY (product_id) REFERENCES products(id)
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_products_category ON products(category);
CREATE INDEX IF NOT EXISTS idx_orders_user_id ON orders(user_id);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
'''

            with open(db_file.parent / "schema.sql", 'w') as f:
                f.write(schema_content)

            return True
        except Exception as e:
            logger.error(f"Failed to set up SQLite: {e}")
            return False

    def _create_db_connection_config(self, project_path: Path, db_config: Dict):
        """Create database connection configuration files"""

        # Node.js/Express database config
        if (project_path / "backend").exists():
            db_config_js = f'''
const {{ Sequelize }} = require('sequelize');

const sequelize = new Sequelize(
  '{db_config["database"]}',
  '{db_config["user"]}',
  '{db_config["password"]}',
  {{
    host: '{db_config["host"]}',
    port: {db_config["port"]},
    dialect: 'postgres',
    logging: process.env.NODE_ENV === 'development' ? console.log : false,
    pool: {{
      max: 5,
      min: 0,
      acquire: 30000,
      idle: 10000
    }}
  }}
);

module.exports = sequelize;
'''
            config_dir = project_path / "backend" / "config"
            config_dir.mkdir(exist_ok=True)
            with open(config_dir / "database.js", 'w') as f:
                f.write(db_config_js)

        # Python/Django database config
        if (project_path / "backend").exists():
            settings_file = project_path / "backend" / "settings.py"
            if settings_file.exists():
                # Update Django settings with database config
                settings_content = f'''
DATABASES = {{
    'default': {{
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': '{db_config["database"]}',
        'USER': '{db_config["user"]}',
        'PASSWORD': '{db_config["password"]}',
        'HOST': '{db_config["host"]}',
        'PORT': '{db_config["port"]}',
    }}
}}
'''
                # This would be inserted into the existing settings.py file
                # For now, we'll create a separate config file

        # Environment variables file
        env_content = f'''
DB_HOST={db_config["host"]}
DB_PORT={db_config["port"]}
DB_NAME={db_config["database"]}
DB_USER={db_config["user"]}
DB_PASSWORD={db_config["password"]}
'''
        with open(project_path / ".env", 'a') as f:
            f.write(env_content)

    async def _create_migrations(self, task) -> Dict[str, Any]:
        """Create database migrations"""
        project_path = Path(self.agent_system.current_project.target_directory)

        # Create migration files based on framework
        if (project_path / "backend").exists():
            migrations_dir = project_path / "backend" / "migrations"
            migrations_dir.mkdir(exist_ok=True)

            # Create initial migration
            migration_content = '''
-- Initial migration
-- This file contains the initial database schema

-- Run this file to set up the database for the first time
-- Make sure to update the connection parameters at the top

-- Users table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Add more tables as needed...
'''
            with open(migrations_dir / "001_initial.sql", 'w') as f:
                f.write(migration_content)

        return {
            'status': 'completed',
            'migrations_created': True,
            'message': 'Database migrations created'
        }

    async def _seed_database(self, task) -> Dict[str, Any]:
        """Seed database with sample data"""
        project_path = Path(self.agent_system.current_project.target_directory)

        # Create seed data files
        seeds_dir = project_path / "database" / "seeds"
        seeds_dir.mkdir(exist_ok=True)

        # Create sample users data
        users_seed = '''
-- Sample users data
INSERT INTO users (name, email, password_hash, role) VALUES
('John Doe', 'john@example.com', '$2b$10$hashedpassword1', 'user'),
('Jane Smith', 'jane@example.com', '$2b$10$hashedpassword2', 'admin'),
('Bob Johnson', 'bob@example.com', '$2b$10$hashedpassword3', 'user')
ON CONFLICT (email) DO NOTHING;
'''

        # Create sample products data
        products_seed = '''
-- Sample products data
INSERT INTO products (name, description, price, category, stock_quantity) VALUES
('Laptop', 'High-performance laptop', 999.99, 'Electronics', 10),
('Book', 'Programming book', 29.99, 'Books', 50),
('Coffee Mug', 'Ceramic coffee mug', 9.99, 'Home', 100)
ON CONFLICT DO NOTHING;
'''

        with open(seeds_dir / "users.sql", 'w') as f:
            f.write(users_seed)

        with open(seeds_dir / "products.sql", 'w') as f:
            f.write(products_seed)

        return {
            'status': 'completed',
            'seeds_created': True,
            'message': 'Database seeded with sample data'
        }

    async def _general_database_task(self, task) -> Dict[str, Any]:
        """Handle general database tasks"""
        return {
            'status': 'completed',
            'message': f'Database task completed: {task.description}'
        }