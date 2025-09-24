"""
Frontend Agent
Specialized in creating beautiful, modern, and functional frontend applications
Supports React, Vue, Svelte, Angular, and Next.js with best practices.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import subprocess

logger = logging.getLogger(__name__)

class FrontendAgent:
    """
    Advanced frontend development agent that creates production-ready
    frontend applications with modern frameworks and best practices.
    """

    def __init__(self, agent_system):
        self.agent_system = agent_system
        self.supported_frameworks = {
            'react': self._create_react_app,
            'vue': self._create_vue_app,
            'svelte': self._create_svelte_app,
            'angular': self._create_angular_app,
            'nextjs': self._create_nextjs_app
        }
        self.component_templates = self._load_component_templates()

    def _load_component_templates(self) -> Dict[str, Any]:
        """Load component templates for different UI elements"""
        templates = {
            'button': self._get_button_template,
            'card': self._get_card_template,
            'form': self._get_form_template,
            'navigation': self._get_navigation_template,
            'dashboard': self._get_dashboard_template,
            'modal': self._get_modal_template
        }
        return templates

    async def execute_task(self, task) -> Dict[str, Any]:
        """Execute a frontend-related task"""
        logger.info(f"Frontend agent executing: {task.description}")

        if 'initialize' in task.description.lower() and 'frontend' in task.description.lower():
            return await self._setup_frontend(task)
        elif 'component' in task.description.lower():
            return await self._create_components(task)
        elif 'ui' in task.description.lower() or 'interface' in task.description.lower():
            return await self._create_ui_layout(task)
        else:
            return await self._general_frontend_task(task)

    async def _setup_frontend(self, task) -> Dict[str, Any]:
        """Set up frontend application based on framework"""
        project_config = self.agent_system.current_project

        if not project_config or not project_config.frontend_framework:
            raise ValueError("No frontend framework specified in project config")

        framework = project_config.frontend_framework.lower()
        project_path = Path(project_config.target_directory)

        if framework not in self.supported_frameworks:
            raise ValueError(f"Unsupported frontend framework: {framework}")

        # Create frontend directory
        frontend_path = project_path / "frontend"
        frontend_path.mkdir(exist_ok=True)

        # Initialize the framework
        success = await self.supported_frameworks[framework](frontend_path)

        if success:
            return {
                'status': 'completed',
                'frontend_path': str(frontend_path),
                'framework': framework,
                'files_created': self._get_created_files(frontend_path)
            }
        else:
            raise Exception(f"Failed to create {framework} application")

    async def _create_react_app(self, project_path: Path) -> bool:
        """Create a React application with modern setup"""
        try:
            # Use create-react-app or Vite for better performance
            subprocess.run([
                'npx', 'create-react-app', '.', '--yes'
            ], cwd=project_path, check=True, capture_output=True)

            # Install additional dependencies for better development
            subprocess.run([
                'npm', 'install', 'axios', 'react-router-dom', 'styled-components',
                '@testing-library/react', '@testing-library/jest-dom'
            ], cwd=project_path, check=True)

            # Create modern folder structure
            self._create_react_structure(project_path)

            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create React app: {e}")
            return False

    async def _create_vue_app(self, project_path: Path) -> bool:
        """Create a Vue.js application"""
        try:
            subprocess.run([
                'npm', 'create', 'vue@latest', '.', '--', '--yes'
            ], cwd=project_path, check=True, capture_output=True)

            # Install additional dependencies
            subprocess.run([
                'npm', 'install', 'axios', 'vue-router', 'pinia'
            ], cwd=project_path, check=True)

            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create Vue app: {e}")
            return False

    async def _create_svelte_app(self, project_path: Path) -> bool:
        """Create a Svelte application"""
        try:
            subprocess.run([
                'npx', 'create-svelte@latest', '.', '--yes'
            ], cwd=project_path, check=True, capture_output=True)

            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create Svelte app: {e}")
            return False

    async def _create_angular_app(self, project_path: Path) -> bool:
        """Create an Angular application"""
        try:
            subprocess.run([
                'npx', '@angular/cli', 'new', '.', '--skip-git', '--yes'
            ], cwd=project_path, check=True, capture_output=True)

            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create Angular app: {e}")
            return False

    async def _create_nextjs_app(self, project_path: Path) -> bool:
        """Create a Next.js application"""
        try:
            subprocess.run([
                'npx', 'create-next-app@latest', '.', '--yes'
            ], cwd=project_path, check=True, capture_output=True)

            # Install additional dependencies
            subprocess.run([
                'npm', 'install', 'axios', 'styled-components'
            ], cwd=project_path, check=True)

            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create Next.js app: {e}")
            return False

    def _create_react_structure(self, project_path: Path):
        """Create modern React folder structure"""
        # Create components directory
        components_dir = project_path / "src" / "components"
        components_dir.mkdir(exist_ok=True)

        # Create pages directory
        pages_dir = project_path / "src" / "pages"
        pages_dir.mkdir(exist_ok=True)

        # Create services directory
        services_dir = project_path / "src" / "services"
        services_dir.mkdir(exist_ok=True)

        # Create utils directory
        utils_dir = project_path / "src" / "utils"
        utils_dir.mkdir(exist_ok=True)

        # Create hooks directory
        hooks_dir = project_path / "src" / "hooks"
        hooks_dir.mkdir(exist_ok=True)

        # Create contexts directory
        contexts_dir = project_path / "src" / "contexts"
        contexts_dir.mkdir(exist_ok=True)

        # Create modern App.js with routing
        self._create_modern_app_structure(project_path)

    def _create_modern_app_structure(self, project_path: Path):
        """Create modern React app structure with routing and state management"""

        # Update App.js for modern structure
        app_js_content = '''
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, CssBaseline } from '@mui/material';
import { createTheme } from '@mui/material/styles';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Login from './pages/Login';
import Register from './pages/Register';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Layout>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />
          </Routes>
        </Layout>
      </Router>
    </ThemeProvider>
  );
}

export default App;
'''

        with open(project_path / "src" / "App.js", 'w') as f:
            f.write(app_js_content)

        # Create Layout component
        self._create_layout_component(project_path)
        self._create_dashboard_page(project_path)
        self._create_auth_pages(project_path)

    def _create_layout_component(self, project_path: Path):
        """Create modern layout component with navigation"""

        layout_content = '''
import React, { useState } from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  IconButton,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Box,
  useMediaQuery,
  useTheme
} from '@mui/material';
import {
  Menu as MenuIcon,
  Dashboard as DashboardIcon,
  AccountCircle,
  ExitToApp
} from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';

const drawerWidth = 240;

function Layout({ children }) {
  const [mobileOpen, setMobileOpen] = useState(false);
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const navigate = useNavigate();
  const location = useLocation();

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const menuItems = [
    { text: 'Dashboard', icon: <DashboardIcon />, path: '/' },
    { text: 'Profile', icon: <AccountCircle />, path: '/profile' }
  ];

  const drawer = (
    <div>
      <Toolbar>
        <Typography variant="h6" noWrap component="div">
          My App
        </Typography>
      </Toolbar>
      <List>
        {menuItems.map((item) => (
          <ListItem key={item.text} disablePadding>
            <ListItemButton
              selected={location.pathname === item.path}
              onClick={() => {
                navigate(item.path);
                if (isMobile) setMobileOpen(false);
              }}
            >
              <ListItemIcon>
                {item.icon}
              </ListItemIcon>
              <ListItemText primary={item.text} />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
    </div>
  );

  return (
    <Box sx={{ display: 'flex' }}>
      <AppBar
        position="fixed"
        sx={{
          width: { md: `calc(100% - ${drawerWidth}px)` },
          ml: { md: `${drawerWidth}px` },
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { md: 'none' } }}
          >
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" noWrap component="div">
            Modern App
          </Typography>
          <Box sx={{ ml: 'auto' }}>
            <Button color="inherit" onClick={() => navigate('/login')}>
              Login
            </Button>
          </Box>
        </Toolbar>
      </AppBar>
      <Box
        component="nav"
        sx={{ width: { md: drawerWidth }, flexShrink: { md: 0 } }}
        aria-label="mailbox folders"
      >
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{
            keepMounted: true, // Better open performance on mobile.
          }}
          sx={{
            display: { xs: 'block', md: 'none' },
            '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
          }}
        >
          {drawer}
        </Drawer>
        <Drawer
          variant="permanent"
          sx={{
            display: { xs: 'none', md: 'block' },
            '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
          }}
          open
        >
          {drawer}
        </Drawer>
      </Box>
      <Box
        component="main"
        sx={{ flexGrow: 1, p: 3, width: { md: `calc(100% - ${drawerWidth}px)` } }}
      >
        <Toolbar />
        {children}
      </Box>
    </Box>
  );
}

export default Layout;
'''

        layout_file = project_path / "src" / "components" / "Layout.js"
        with open(layout_file, 'w') as f:
            f.write(layout_content)

    def _create_dashboard_page(self, project_path: Path):
        """Create dashboard page component"""

        dashboard_content = '''
import React from 'react';
import {
  Container,
  Grid,
  Paper,
  Typography,
  Box,
  Card,
  CardContent
} from '@mui/material';

function Dashboard() {
  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Paper sx={{ p: 2, display: 'flex', flexDirection: 'column' }}>
            <Typography component="h1" variant="h4" color="primary">
              Dashboard
            </Typography>
            <Typography variant="body1" color="text.secondary" sx={{ mt: 1 }}>
              Welcome to your modern dashboard application
            </Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} md={8} lg={9}>
          <Card>
            <CardContent>
              <Typography variant="h5" component="h2">
                Main Content Area
              </Typography>
              <Typography variant="body2" component="p" sx={{ mt: 2 }}>
                This is where your main content would go. You can add charts,
                tables, forms, or any other components here.
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={4} lg={3}>
          <Card>
            <CardContent>
              <Typography variant="h5" component="h2">
                Sidebar
              </Typography>
              <Typography variant="body2" component="p" sx={{ mt: 2 }}>
                Additional information or widgets can go here.
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Container>
  );
}

export default Dashboard;
'''

        dashboard_file = project_path / "src" / "pages" / "Dashboard.js"
        with open(dashboard_file, 'w') as f:
            f.write(dashboard_content)

    def _create_auth_pages(self, project_path: Path):
        """Create authentication pages"""

        # Login page
        login_content = '''
import React, { useState } from 'react';
import {
  Container,
  Paper,
  TextField,
  Button,
  Typography,
  Box,
  Alert,
  Link
} from '@mui/material';
import { useNavigate } from 'react-router-dom';

function Login() {
  const [formData, setFormData] = useState({
    email: '',
    password: ''
  });
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      // Add your authentication logic here
      console.log('Login attempt:', formData);
      navigate('/');
    } catch (err) {
      setError('Invalid credentials');
    }
  };

  return (
    <Container component="main" maxWidth="xs">
      <Box
        sx={{
          marginTop: 8,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
        }}
      >
        <Paper elevation={3} sx={{ padding: 4, width: '100%' }}>
          <Typography component="h1" variant="h5" align="center">
            Sign In
          </Typography>
          {error && <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>}
          <Box component="form" onSubmit={handleSubmit} sx={{ mt: 1 }}>
            <TextField
              margin="normal"
              required
              fullWidth
              id="email"
              label="Email Address"
              name="email"
              autoComplete="email"
              autoFocus
              value={formData.email}
              onChange={handleChange}
            />
            <TextField
              margin="normal"
              required
              fullWidth
              name="password"
              label="Password"
              type="password"
              id="password"
              autoComplete="current-password"
              value={formData.password}
              onChange={handleChange}
            />
            <Button
              type="submit"
              fullWidth
              variant="contained"
              sx={{ mt: 3, mb: 2 }}
            >
              Sign In
            </Button>
            <Link href="/register" variant="body2" sx={{ display: 'block', textAlign: 'center' }}>
              Don't have an account? Sign Up
            </Link>
          </Box>
        </Paper>
      </Box>
    </Container>
  );
}

export default Login;
'''

        login_file = project_path / "src" / "pages" / "Login.js"
        with open(login_file, 'w') as f:
            f.write(login_content)

        # Register page
        register_content = '''
import React, { useState } from 'react';
import {
  Container,
  Paper,
  TextField,
  Button,
  Typography,
  Box,
  Alert,
  Link
} from '@mui/material';
import { useNavigate } from 'react-router-dom';

function Register() {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: '',
    confirmPassword: ''
  });
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (formData.password !== formData.confirmPassword) {
      setError('Passwords do not match');
      return;
    }
    try {
      // Add your registration logic here
      console.log('Registration attempt:', formData);
      navigate('/');
    } catch (err) {
      setError('Registration failed');
    }
  };

  return (
    <Container component="main" maxWidth="xs">
      <Box
        sx={{
          marginTop: 8,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
        }}
      >
        <Paper elevation={3} sx={{ padding: 4, width: '100%' }}>
          <Typography component="h1" variant="h5" align="center">
            Sign Up
          </Typography>
          {error && <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>}
          <Box component="form" onSubmit={handleSubmit} sx={{ mt: 1 }}>
            <TextField
              margin="normal"
              required
              fullWidth
              id="name"
              label="Full Name"
              name="name"
              autoComplete="name"
              autoFocus
              value={formData.name}
              onChange={handleChange}
            />
            <TextField
              margin="normal"
              required
              fullWidth
              id="email"
              label="Email Address"
              name="email"
              autoComplete="email"
              value={formData.email}
              onChange={handleChange}
            />
            <TextField
              margin="normal"
              required
              fullWidth
              name="password"
              label="Password"
              type="password"
              id="password"
              autoComplete="new-password"
              value={formData.password}
              onChange={handleChange}
            />
            <TextField
              margin="normal"
              required
              fullWidth
              name="confirmPassword"
              label="Confirm Password"
              type="password"
              id="confirmPassword"
              value={formData.confirmPassword}
              onChange={handleChange}
            />
            <Button
              type="submit"
              fullWidth
              variant="contained"
              sx={{ mt: 3, mb: 2 }}
            >
              Sign Up
            </Button>
            <Link href="/login" variant="body2" sx={{ display: 'block', textAlign: 'center' }}>
              Already have an account? Sign In
            </Link>
          </Box>
        </Paper>
      </Box>
    </Container>
  );
}

export default Register;
'''

        register_file = project_path / "src" / "pages" / "Register.js"
        with open(register_file, 'w') as f:
            f.write(register_content)

    async def _create_components(self, task) -> Dict[str, Any]:
        """Create reusable UI components"""
        project_config = self.agent_system.current_project
        project_path = Path(project_config.target_directory) / "frontend"

        components = []
        if 'button' in task.description.lower():
            components.append('button')
        if 'card' in task.description.lower():
            components.append('card')
        if 'form' in task.description.lower():
            components.append('form')
        if 'navigation' in task.description.lower():
            components.append('navigation')

        created_files = []
        for component in components:
            if component in self.component_templates:
                template_func = self.component_templates[component]
                files = template_func(project_path)
                created_files.extend(files)

        return {
            'status': 'completed',
            'components_created': components,
            'files_created': created_files
        }

    async def _create_ui_layout(self, task) -> Dict[str, Any]:
        """Create UI layout and styling"""
        project_config = self.agent_system.current_project
        project_path = Path(project_config.target_directory) / "frontend"

        # Create modern styling with CSS-in-JS or styled-components
        styling_content = '''
// Modern theme configuration
export const theme = {
  colors: {
    primary: '#1976d2',
    secondary: '#dc004e',
    success: '#4caf50',
    warning: '#ff9800',
    error: '#f44336',
    background: '#f5f5f5',
    surface: '#ffffff',
    text: '#333333',
    textSecondary: '#666666'
  },
  spacing: {
    xs: 4,
    sm: 8,
    md: 16,
    lg: 24,
    xl: 32
  },
  breakpoints: {
    xs: 0,
    sm: 600,
    md: 960,
    lg: 1280,
    xl: 1920
  }
};

// Responsive utility functions
export const isMobile = (width) => width < theme.breakpoints.sm;
export const isTablet = (width) => width >= theme.breakpoints.sm && width < theme.breakpoints.md;
export const isDesktop = (width) => width >= theme.breakpoints.md;
'''

        styling_file = project_path / "src" / "theme.js"
        with open(styling_file, 'w') as f:
            f.write(styling_content)

        return {
            'status': 'completed',
            'layout_created': 'modern_theme',
            'files_created': [str(styling_file)]
        }

    async def _general_frontend_task(self, task) -> Dict[str, Any]:
        """Handle general frontend tasks"""
        return {
            'status': 'completed',
            'message': f'Frontend task completed: {task.description}'
        }

    def _get_created_files(self, directory: Path) -> List[str]:
        """Get list of files created in a directory"""
        files = []
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                files.append(str(file_path.relative_to(directory)))
        return files

    # Template methods for components
    def _get_button_template(self, project_path: Path) -> List[str]:
        """Get button component template"""
        button_content = '''
import React from 'react';
import { Button as MuiButton } from '@mui/material';
import { styled } from '@mui/material/styles';

const StyledButton = styled(MuiButton)(({ theme, variant }) => ({
  borderRadius: 8,
  textTransform: 'none',
  fontWeight: 600,
  padding: '12px 24px',
  ...(variant === 'primary' && {
    backgroundColor: theme.colors.primary,
    color: 'white',
    '&:hover': {
      backgroundColor: '#1565c0'
    }
  }),
  ...(variant === 'secondary' && {
    backgroundColor: 'transparent',
    color: theme.colors.primary,
    border: `2px solid ${theme.colors.primary}`,
    '&:hover': {
      backgroundColor: theme.colors.primary,
      color: 'white'
    }
  })
}));

function Button({ children, variant = 'primary', ...props }) {
  return (
    <StyledButton variant={variant} {...props}>
      {children}
    </StyledButton>
  );
}

export default Button;
'''
        button_file = project_path / "src" / "components" / "Button.js"
        with open(button_file, 'w') as f:
            f.write(button_content)

        return [str(button_file.relative_to(project_path))]

    def _get_card_template(self, project_path: Path) -> List[str]:
        """Get card component template"""
        card_content = '''
import React from 'react';
import { Card as MuiCard, CardContent, CardActions, Typography } from '@mui/material';
import { styled } from '@mui/material/styles';

const StyledCard = styled(MuiCard)(({ theme }) => ({
  borderRadius: 12,
  boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
  transition: 'all 0.3s ease',
  '&:hover': {
    transform: 'translateY(-4px)',
    boxShadow: '0 8px 25px rgba(0, 0, 0, 0.15)'
  }
}));

function Card({ title, children, actions, ...props }) {
  return (
    <StyledCard {...props}>
      {title && (
        <CardContent>
          <Typography variant="h6" component="h2">
            {title}
          </Typography>
        </CardContent>
      )}
      <CardContent>
        {children}
      </CardContent>
      {actions && (
        <CardActions>
          {actions}
        </CardActions>
      )}
    </StyledCard>
  );
}

export default Card;
'''
        card_file = project_path / "src" / "components" / "Card.js"
        with open(card_file, 'w') as f:
            f.write(card_content)

        return [str(card_file.relative_to(project_path))]

    def _get_form_template(self, project_path: Path) -> List[str]:
        """Get form component template"""
        form_content = '''
import React, { useState } from 'react';
import { TextField, Button, Box, Typography } from '@mui/material';
import { styled } from '@mui/material/styles';

const FormContainer = styled(Box)(({ theme }) => ({
  maxWidth: 400,
  margin: '0 auto',
  padding: 24,
  borderRadius: 12,
  backgroundColor: 'white',
  boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)'
}));

function Form({ fields, onSubmit, title, submitText = 'Submit' }) {
  const [formData, setFormData] = useState({});
  const [errors, setErrors] = useState({});

  const handleChange = (field) => (event) => {
    setFormData({
      ...formData,
      [field]: event.target.value
    });
    // Clear error when user starts typing
    if (errors[field]) {
      setErrors({
        ...errors,
        [field]: ''
      });
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(formData);
  };

  return (
    <FormContainer>
      {title && (
        <Typography variant="h5" component="h1" align="center" gutterBottom>
          {title}
        </Typography>
      )}
      <Box component="form" onSubmit={handleSubmit}>
        {fields.map((field) => (
          <TextField
            key={field.name}
            margin="normal"
            required={field.required}
            fullWidth
            label={field.label}
            type={field.type || 'text'}
            value={formData[field.name] || ''}
            onChange={handleChange(field.name)}
            error={!!errors[field.name]}
            helperText={errors[field.name]}
            {...field.props}
          />
        ))}
        <Button
          type="submit"
          fullWidth
          variant="contained"
          sx={{ mt: 3, mb: 2 }}
        >
          {submitText}
        </Button>
      </Box>
    </FormContainer>
  );
}

export default Form;
'''
        form_file = project_path / "src" / "components" / "Form.js"
        with open(form_file, 'w') as f:
            f.write(form_content)

        return [str(form_file.relative_to(project_path))]

    def _get_navigation_template(self, project_path: Path) -> List[str]:
        """Get navigation component template"""
        nav_content = '''
import React, { useState } from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  IconButton,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Box,
  useMediaQuery,
  useTheme
} from '@mui/material';
import { Menu as MenuIcon, Dashboard, AccountCircle } from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';

function Navigation({ items = [] }) {
  const [mobileOpen, setMobileOpen] = useState(false);
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const navigate = useNavigate();
  const location = useLocation();

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const defaultItems = [
    { text: 'Dashboard', icon: <Dashboard />, path: '/' },
    { text: 'Profile', icon: <AccountCircle />, path: '/profile' }
  ];

  const navItems = items.length > 0 ? items : defaultItems;

  const drawer = (
    <div>
      <Toolbar>
        <Typography variant="h6" noWrap component="div">
          Navigation
        </Typography>
      </Toolbar>
      <List>
        {navItems.map((item) => (
          <ListItem key={item.text} disablePadding>
            <ListItemButton
              selected={location.pathname === item.path}
              onClick={() => {
                navigate(item.path);
                if (isMobile) setMobileOpen(false);
              }}
            >
              <ListItemIcon>
                {item.icon}
              </ListItemIcon>
              <ListItemText primary={item.text} />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
    </div>
  );

  return (
    <>
      <AppBar position="static">
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { md: 'none' } }}
          >
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
            My App
          </Typography>
          <Box sx={{ display: { xs: 'none', md: 'flex' } }}>
            {navItems.map((item) => (
              <Button
                key={item.text}
                color="inherit"
                onClick={() => navigate(item.path)}
                sx={{
                  color: location.pathname === item.path ? 'secondary.main' : 'inherit'
                }}
              >
                {item.text}
              </Button>
            ))}
          </Box>
        </Toolbar>
      </AppBar>
      <Box component="nav">
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{
            keepMounted: true,
          }}
          sx={{
            display: { xs: 'block', md: 'none' },
            '& .MuiDrawer-paper': { boxSizing: 'border-box', width: 240 },
          }}
        >
          {drawer}
        </Drawer>
      </Box>
    </>
  );
}

export default Navigation;
'''
        nav_file = project_path / "src" / "components" / "Navigation.js"
        with open(nav_file, 'w') as f:
            f.write(nav_content)

        return [str(nav_file.relative_to(project_path))]

    def _get_dashboard_template(self, project_path: Path) -> List[str]:
        """Get dashboard component template"""
        dashboard_content = '''
import React from 'react';
import {
  Container,
  Grid,
  Paper,
  Typography,
  Box,
  Card,
  CardContent,
  CardHeader
} from '@mui/material';

function Dashboard({ widgets = [] }) {
  const defaultWidgets = [
    {
      title: 'Welcome',
      content: 'Welcome to your dashboard! This is where you can manage your application.'
    },
    {
      title: 'Statistics',
      content: 'Your statistics and metrics will appear here.'
    }
  ];

  const displayWidgets = widgets.length > 0 ? widgets : defaultWidgets;

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Grid container spacing={3}>
        {displayWidgets.map((widget, index) => (
          <Grid item xs={12} md={6} lg={4} key={index}>
            <Card>
              <CardHeader title={widget.title} />
              <CardContent>
                <Typography variant="body2" color="text.secondary">
                  {widget.content}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Container>
  );
}

export default Dashboard;
'''
        dashboard_file = project_path / "src" / "components" / "Dashboard.js"
        with open(dashboard_file, 'w') as f:
            f.write(dashboard_content)

        return [str(dashboard_file.relative_to(project_path))]

    def _get_modal_template(self, project_path: Path) -> List[str]:
        """Get modal component template"""
        modal_content = '''
import React from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  IconButton
} from '@mui/material';
import { Close as CloseIcon } from '@mui/icons-material';

function Modal({ open, onClose, title, children, actions }) {
  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>
        {title}
        <IconButton
          aria-label="close"
          onClick={onClose}
          sx={{
            position: 'absolute',
            right: 8,
            top: 8,
            color: (theme) => theme.palette.grey[500],
          }}
        >
          <CloseIcon />
        </IconButton>
      </DialogTitle>
      <DialogContent dividers>
        {children}
      </DialogContent>
      {actions && (
        <DialogActions>
          {actions}
        </DialogActions>
      )}
    </Dialog>
  );
}

export default Modal;
'''
        modal_file = project_path / "src" / "components" / "Modal.js"
        with open(modal_file, 'w') as f:
            f.write(modal_content)

        return [str(modal_file.relative_to(project_path))]