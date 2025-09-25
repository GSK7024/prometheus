#!/usr/bin/env python3
"""
WEB APPLICATION - TDD IMPLEMENTATION EXAMPLE
Demonstrates pure TDD-first development for web applications
"""

import unittest
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
import re
from datetime import datetime

@dataclass
class User:
    """User model for web application"""
    id: int
    username: str
    email: str
    password_hash: str
    created_at: datetime
    is_active: bool = True

@dataclass
class Post:
    """Post model for web application"""
    id: int
    title: str
    content: str
    author_id: int
    created_at: datetime
    updated_at: datetime
    published: bool = True

class WebApplicationResult:
    """Result structure for web application operations"""

    def __init__(self, success: bool, data: Any, message: str, metadata: Dict[str, Any] = None):
        self.success = success
        self.data = data
        self.message = message
        self.metadata = metadata or {}

class UserAuthenticationService:
    """
    User authentication service
    Built using pure Test-Driven Development methodology
    """

    def __init__(self):
        """Initialize the authentication service"""
        self.users: Dict[str, User] = {}
        self.user_counter = 0
        self.config = self._load_configuration()

    def _load_configuration(self) -> Dict[str, Any]:
        """Load configuration settings"""
        return {
            'password_min_length': 8,
            'username_pattern': r'^[a-zA-Z0-9_]{3,20}$',
            'email_pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'max_login_attempts': 3,
            'lockout_duration_minutes': 15
        }

    def validate_input(self, input_data: Any) -> WebApplicationResult:
        """Validate input parameters"""
        if input_data is None:
            return WebApplicationResult(
                success=False,
                data=None,
                message="Input data cannot be None",
                metadata={'error_type': 'null_input'}
            )

        if not isinstance(input_data, dict):
            return WebApplicationResult(
                success=False,
                data=None,
                message="Input must be a dictionary",
                metadata={'error_type': 'invalid_type'}
            )

        return WebApplicationResult(
            success=True,
            data=input_data,
            message="Input validation passed"
        )

    def validate_username(self, username: str) -> WebApplicationResult:
        """Validate username format and availability"""
        config = self.config

        if not isinstance(username, str):
            return WebApplicationResult(
                success=False,
                data=None,
                message="Username must be a string",
                metadata={'error_type': 'invalid_type'}
            )

        if not re.match(config['username_pattern'], username):
            return WebApplicationResult(
                success=False,
                data=None,
                message="Username must be 3-20 characters, alphanumeric and underscores only",
                metadata={'error_type': 'invalid_format'}
            )

        if username in self.users:
            return WebApplicationResult(
                success=False,
                data=None,
                message="Username already exists",
                metadata={'error_type': 'username_taken'}
            )

        return WebApplicationResult(
            success=True,
            data=username,
            message="Username is valid and available"
        )

    def validate_email(self, email: str) -> WebApplicationResult:
        """Validate email format"""
        config = self.config

        if not isinstance(email, str):
            return WebApplicationResult(
                success=False,
                data=None,
                message="Email must be a string",
                metadata={'error_type': 'invalid_type'}
            )

        if not re.match(config['email_pattern'], email):
            return WebApplicationResult(
                success=False,
                data=None,
                message="Invalid email format",
                metadata={'error_type': 'invalid_format'}
            )

        return WebApplicationResult(
            success=True,
            data=email,
            message="Email format is valid"
        )

    def validate_password(self, password: str) -> WebApplicationResult:
        """Validate password strength"""
        config = self.config

        if not isinstance(password, str):
            return WebApplicationResult(
                success=False,
                data=None,
                message="Password must be a string",
                metadata={'error_type': 'invalid_type'}
            )

        if len(password) < config['password_min_length']:
            return WebApplicationResult(
                success=False,
                data=None,
                message=f"Password must be at least {config['password_min_length']} characters long",
                metadata={'error_type': 'password_too_short'}
            )

        # Check for common passwords (simplified)
        common_passwords = ['password', '12345678', 'qwerty123', 'admin123']
        if password.lower() in common_passwords:
            return WebApplicationResult(
                success=False,
                data=None,
                message="Password is too common",
                metadata={'error_type': 'common_password'}
            )

        return WebApplicationResult(
            success=True,
            data=password,
            message="Password meets security requirements"
        )

    def hash_password(self, password: str) -> str:
        """Hash password (simplified for demo)"""
        # In real implementation, use bcrypt or similar
        return f"hashed_{password}"

    def register_user(self, username: str, email: str, password: str) -> WebApplicationResult:
        """Register a new user"""
        try:
            # Validate all inputs
            username_validation = self.validate_username(username)
            if not username_validation.success:
                return username_validation

            email_validation = self.validate_email(email)
            if not email_validation.success:
                return email_validation

            password_validation = self.validate_password(password)
            if not password_validation.success:
                return password_validation

            # Create user
            self.user_counter += 1
            user = User(
                id=self.user_counter,
                username=username,
                email=email,
                password_hash=self.hash_password(password),
                created_at=datetime.now()
            )

            self.users[username] = user

            return WebApplicationResult(
                success=True,
                data=user,
                message="User registered successfully",
                metadata={'user_id': user.id}
            )

        except Exception as e:
            return WebApplicationResult(
                success=False,
                data=None,
                message=f"Error registering user: {str(e)}",
                metadata={'error_type': 'registration_error'}
            )

    def authenticate_user(self, username: str, password: str) -> WebApplicationResult:
        """Authenticate user credentials"""
        try:
            if username not in self.users:
                return WebApplicationResult(
                    success=False,
                    data=None,
                    message="User not found",
                    metadata={'error_type': 'user_not_found'}
                )

            user = self.users[username]

            if not user.is_active:
                return WebApplicationResult(
                    success=False,
                    data=None,
                    message="Account is disabled",
                    metadata={'error_type': 'account_disabled'}
                )

            # Verify password (simplified)
            if user.password_hash != self.hash_password(password):
                return WebApplicationResult(
                    success=False,
                    data=None,
                    message="Invalid password",
                    metadata={'error_type': 'invalid_password'}
                )

            return WebApplicationResult(
                success=True,
                data=user,
                message="Authentication successful",
                metadata={'user_id': user.id}
            )

        except Exception as e:
            return WebApplicationResult(
                success=False,
                data=None,
                message=f"Error authenticating user: {str(e)}",
                metadata={'error_type': 'authentication_error'}
            )

    def get_user_by_id(self, user_id: int) -> WebApplicationResult:
        """Get user by ID"""
        try:
            for user in self.users.values():
                if user.id == user_id:
                    return WebApplicationResult(
                        success=True,
                        data=user,
                        message="User found",
                        metadata={'user_id': user.id}
                    )

            return WebApplicationResult(
                success=False,
                data=None,
                message="User not found",
                metadata={'error_type': 'user_not_found'}
            )

        except Exception as e:
            return WebApplicationResult(
                success=False,
                data=None,
                message=f"Error retrieving user: {str(e)}",
                metadata={'error_type': 'database_error'}
            )

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and health information"""
        return {
            'status': 'operational',
            'domain': 'web_authentication',
            'version': '2.0.0',
            'total_users': len(self.users),
            'supported_operations': [
                'user_registration',
                'user_authentication',
                'password_validation',
                'email_validation',
                'username_validation'
            ],
            'security_features': [
                'password_strength_validation',
                'common_password_protection',
                'input_sanitization'
            ]
        }

class PostManagementService:
    """
    Post management service
    Built using pure Test-Driven Development methodology
    """

    def __init__(self, auth_service: UserAuthenticationService):
        """Initialize the post management service"""
        self.auth_service = auth_service
        self.posts: Dict[int, Post] = {}
        self.post_counter = 0

    def validate_post_data(self, title: str, content: str, author_id: int) -> WebApplicationResult:
        """Validate post creation data"""
        if not isinstance(title, str) or not isinstance(content, str):
            return WebApplicationResult(
                success=False,
                data=None,
                message="Title and content must be strings",
                metadata={'error_type': 'invalid_type'}
            )

        if len(title.strip()) < 3:
            return WebApplicationResult(
                success=False,
                data=None,
                message="Title must be at least 3 characters long",
                metadata={'error_type': 'title_too_short'}
            )

        if len(title) > 200:
            return WebApplicationResult(
                success=False,
                data=None,
                message="Title must be less than 200 characters",
                metadata={'error_type': 'title_too_long'}
            )

        if len(content.strip()) < 10:
            return WebApplicationResult(
                success=False,
                data=None,
                message="Content must be at least 10 characters long",
                metadata={'error_type': 'content_too_short'}
            )

        # Check if author exists
        author_result = self.auth_service.get_user_by_id(author_id)
        if not author_result.success:
            return WebApplicationResult(
                success=False,
                data=None,
                message="Author not found",
                metadata={'error_type': 'invalid_author'}
            )

        return WebApplicationResult(
            success=True,
            data={'title': title, 'content': content, 'author_id': author_id},
            message="Post data is valid"
        )

    def create_post(self, title: str, content: str, author_id: int) -> WebApplicationResult:
        """Create a new post"""
        try:
            # Validate input data
            validation = self.validate_post_data(title, content, author_id)
            if not validation.success:
                return validation

            # Create post
            self.post_counter += 1
            post = Post(
                id=self.post_counter,
                title=title.strip(),
                content=content.strip(),
                author_id=author_id,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )

            self.posts[post.id] = post

            return WebApplicationResult(
                success=True,
                data=post,
                message="Post created successfully",
                metadata={'post_id': post.id}
            )

        except Exception as e:
            return WebApplicationResult(
                success=False,
                data=None,
                message=f"Error creating post: {str(e)}",
                metadata={'error_type': 'creation_error'}
            )

    def get_post_by_id(self, post_id: int) -> WebApplicationResult:
        """Get post by ID"""
        try:
            if post_id not in self.posts:
                return WebApplicationResult(
                    success=False,
                    data=None,
                    message="Post not found",
                    metadata={'error_type': 'post_not_found'}
                )

            post = self.posts[post_id]
            return WebApplicationResult(
                success=True,
                data=post,
                message="Post retrieved successfully"
            )

        except Exception as e:
            return WebApplicationResult(
                success=False,
                data=None,
                message=f"Error retrieving post: {str(e)}",
                metadata={'error_type': 'retrieval_error'}
            )

    def get_posts_by_author(self, author_id: int) -> WebApplicationResult:
        """Get all posts by an author"""
        try:
            author_posts = [
                post for post in self.posts.values()
                if post.author_id == author_id
            ]

            return WebApplicationResult(
                success=True,
                data=author_posts,
                message=f"Found {len(author_posts)} posts by author"
            )

        except Exception as e:
            return WebApplicationResult(
                success=False,
                data=None,
                message=f"Error retrieving posts: {str(e)}",
                metadata={'error_type': 'retrieval_error'}
            )

# Factory functions
def create_user_authentication_service() -> UserAuthenticationService:
    """Factory function for user authentication service"""
    return UserAuthenticationService()

def create_post_management_service(auth_service: UserAuthenticationService) -> PostManagementService:
    """Factory function for post management service"""
    return PostManagementService(auth_service)

# Comprehensive test suite demonstrating TDD approach
class TestUserAuthenticationService(unittest.TestCase):
    """Test suite for user authentication service"""

    def setUp(self):
        """Set up test fixtures"""
        self.auth_service = create_user_authentication_service()

    def tearDown(self):
        """Clean up after tests"""
        pass

    # Unit Tests
    def test_username_validation(self):
        """Test username validation"""
        # Valid usernames
        valid_usernames = ['user123', 'test_user', 'UserName', 'a_b_c']
        for username in valid_usernames:
            result = self.auth_service.validate_username(username)
            self.assertTrue(result.success, f"Username {username} should be valid")

        # Invalid usernames
        invalid_usernames = ['ab', 'a' * 21, 'user-name', 'user.name', 'user name']
        for username in invalid_usernames:
            result = self.auth_service.validate_username(username)
            self.assertFalse(result.success, f"Username {username} should be invalid")

    def test_email_validation(self):
        """Test email validation"""
        # Valid emails
        valid_emails = ['user@example.com', 'test.user@domain.org', 'user123@test.co.uk']
        for email in valid_emails:
            result = self.auth_service.validate_email(email)
            self.assertTrue(result.success, f"Email {email} should be valid")

        # Invalid emails
        invalid_emails = ['invalid-email', '@example.com', 'user@', 'user.example.com']
        for email in invalid_emails:
            result = self.auth_service.validate_email(email)
            self.assertFalse(result.success, f"Email {email} should be invalid")

    def test_password_validation(self):
        """Test password validation"""
        # Valid passwords
        valid_passwords = ['password123', 'SecurePass123', 'MyPassword!@#']
        for password in valid_passwords:
            result = self.auth_service.validate_password(password)
            self.assertTrue(result.success, f"Password {password} should be valid")

        # Invalid passwords
        invalid_passwords = ['1234567', 'password', 'qwerty123', 'a' * 5]
        for password in invalid_passwords:
            result = self.auth_service.validate_password(password)
            self.assertFalse(result.success, f"Password {password} should be invalid")

    def test_user_registration(self):
        """Test user registration"""
        result = self.auth_service.register_user('testuser', 'test@example.com', 'password123')
        self.assertTrue(result.success)
        self.assertEqual(result.data.username, 'testuser')
        self.assertEqual(result.data.email, 'test@example.com')

        # Test duplicate username
        result = self.auth_service.register_user('testuser', 'another@example.com', 'password456')
        self.assertFalse(result.success)
        self.assertIn('already exists', result.message)

    def test_user_authentication(self):
        """Test user authentication"""
        # Register user first
        self.auth_service.register_user('authuser', 'auth@example.com', 'authpass123')

        # Test valid authentication
        result = self.auth_service.authenticate_user('authuser', 'authpass123')
        self.assertTrue(result.success)
        self.assertEqual(result.data.username, 'authuser')

        # Test invalid password
        result = self.auth_service.authenticate_user('authuser', 'wrongpass')
        self.assertFalse(result.success)
        self.assertIn('Invalid password', result.message)

        # Test non-existent user
        result = self.auth_service.authenticate_user('nonexistent', 'password')
        self.assertFalse(result.success)
        self.assertIn('not found', result.message.lower())

    # Integration Tests
    def test_complete_user_workflow(self):
        """Test complete user registration and authentication workflow"""
        username = 'integration_user'
        email = 'integration@example.com'
        password = 'integration123'

        # Register user
        reg_result = self.auth_service.register_user(username, email, password)
        self.assertTrue(reg_result.success)

        # Authenticate user
        auth_result = self.auth_service.authenticate_user(username, password)
        self.assertTrue(auth_result.success)

        # Verify user data consistency
        self.assertEqual(reg_result.data.id, auth_result.data.id)
        self.assertEqual(reg_result.data.username, auth_result.data.username)

    # Edge Case Tests
    def test_empty_inputs(self):
        """Test handling of empty inputs"""
        result = self.auth_service.validate_input("")
        self.assertTrue(result.success)  # Empty string is valid input

        result = self.auth_service.validate_input(None)
        self.assertFalse(result.success)
        self.assertIn('cannot be None', result.message)

    def test_boundary_conditions(self):
        """Test boundary conditions"""
        # Test username length limits
        result = self.auth_service.validate_username('ab')  # Too short
        self.assertFalse(result.success)

        result = self.auth_service.validate_username('a' * 20)  # Maximum length
        self.assertTrue(result.success)

        result = self.auth_service.validate_username('a' * 21)  # Too long
        self.assertFalse(result.success)

    def test_concurrent_users(self):
        """Test handling of concurrent user registrations"""
        users = [
            ('user1', 'user1@example.com', 'pass1'),
            ('user2', 'user2@example.com', 'pass2'),
            ('user3', 'user3@example.com', 'pass3')
        ]

        results = []
        for user_data in users:
            result = self.auth_service.register_user(*user_data)
            results.append(result)
            self.assertTrue(result.success, f"Failed to register {user_data[0]}")

        # All should succeed
        self.assertEqual(len(results), 3)
        self.assertEqual(self.auth_service.user_counter, 3)

class TestPostManagementService(unittest.TestCase):
    """Test suite for post management service"""

    def setUp(self):
        """Set up test fixtures"""
        self.auth_service = create_user_authentication_service()
        self.post_service = create_post_management_service(self.auth_service)

        # Create test user
        self.auth_service.register_user('postuser', 'post@example.com', 'postpass123')

    def tearDown(self):
        """Clean up after tests"""
        pass

    # Unit Tests
    def test_post_creation(self):
        """Test post creation"""
        result = self.post_service.create_post('Test Post', 'This is test content', 1)
        self.assertTrue(result.success)
        self.assertEqual(result.data.title, 'Test Post')
        self.assertEqual(result.data.author_id, 1)

    def test_post_validation(self):
        """Test post data validation"""
        # Valid post data
        result = self.post_service.validate_post_data('Valid Title', 'Valid content that is long enough', 1)
        self.assertTrue(result.success)

        # Invalid title (too short)
        result = self.post_service.validate_post_data('ab', 'Valid content', 1)
        self.assertFalse(result.success)
        self.assertIn('at least 3 characters', result.message)

        # Invalid title (too long)
        long_title = 'a' * 201
        result = self.post_service.validate_post_data(long_title, 'Valid content', 1)
        self.assertFalse(result.success)
        self.assertIn('less than 200 characters', result.message)

        # Invalid content (too short)
        result = self.post_service.validate_post_data('Valid Title', 'short', 1)
        self.assertFalse(result.success)
        self.assertIn('at least 10 characters', result.message)

    def test_post_retrieval(self):
        """Test post retrieval by ID"""
        # Create a post
        create_result = self.post_service.create_post('Test Post', 'Test content', 1)
        post_id = create_result.data.id

        # Retrieve the post
        result = self.post_service.get_post_by_id(post_id)
        self.assertTrue(result.success)
        self.assertEqual(result.data.title, 'Test Post')

        # Try to retrieve non-existent post
        result = self.post_service.get_post_by_id(999)
        self.assertFalse(result.success)
        self.assertIn('not found', result.message.lower())

    def test_posts_by_author(self):
        """Test retrieving posts by author"""
        # Create multiple posts
        self.post_service.create_post('Post 1', 'Content 1', 1)
        self.post_service.create_post('Post 2', 'Content 2', 1)
        self.post_service.create_post('Post 3', 'Content 3', 1)

        # Get posts by author
        result = self.post_service.get_posts_by_author(1)
        self.assertTrue(result.success)
        self.assertEqual(len(result.data), 3)

    # Integration Tests
    def test_complete_post_workflow(self):
        """Test complete post creation and retrieval workflow"""
        # Create post
        create_result = self.post_service.create_post('Workflow Test', 'Test content for workflow', 1)
        self.assertTrue(create_result.success)
        post_id = create_result.data.id

        # Retrieve post
        retrieve_result = self.post_service.get_post_by_id(post_id)
        self.assertTrue(retrieve_result.success)

        # Verify data consistency
        self.assertEqual(create_result.data.title, retrieve_result.data.title)
        self.assertEqual(create_result.data.content, retrieve_result.data.content)
        self.assertEqual(create_result.data.author_id, retrieve_result.data.author_id)

    # Edge Case Tests
    def test_invalid_author_id(self):
        """Test post creation with invalid author ID"""
        result = self.post_service.create_post('Test Post', 'Test content', 999)
        self.assertFalse(result.success)
        self.assertIn('Author not found', result.message)

    def test_empty_post_data(self):
        """Test post creation with empty data"""
        result = self.post_service.create_post('', '', 1)
        self.assertFalse(result.success)
        self.assertIn('at least 3 characters', result.message)

if __name__ == "__main__":
    # Run comprehensive test suite
    unittest.main(verbosity=2)