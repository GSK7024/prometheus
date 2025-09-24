"""
UI/UX Agent
Specialized in creating beautiful, modern, and user-friendly interfaces
with excellent user experience design principles.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class UIUXAgent:
    """
    Advanced UI/UX design agent that creates beautiful, modern interfaces
    with excellent user experience and accessibility features.
    """

    def __init__(self, agent_system):
        self.agent_system = agent_system
        self.design_principles = {
            'simplicity': 'Keep interfaces simple and uncluttered',
            'consistency': 'Maintain consistent design patterns',
            'accessibility': 'Ensure WCAG compliance',
            'responsiveness': 'Design for all device sizes',
            'performance': 'Optimize for fast loading'
        }

    async def execute_task(self, task) -> Dict[str, Any]:
        """Execute a UI/UX-related task"""
        logger.info(f"UI/UX agent executing: {task.description}")

        if 'design' in task.description.lower() or 'ui' in task.description.lower():
            return await self._create_design_system(task)
        elif 'responsive' in task.description.lower():
            return await self._implement_responsive_design(task)
        elif 'accessibility' in task.description.lower():
            return await self._implement_accessibility(task)
        elif 'theme' in task.description.lower():
            return await self._create_theme(task)
        else:
            return await self._general_ui_task(task)

    async def _create_design_system(self, task) -> Dict[str, Any]:
        """Create a comprehensive design system"""
        project_config = self.agent_system.current_project
        project_path = Path(project_config.target_directory) / "frontend"

        # Create design system structure
        design_system = {
            'colors': self._create_color_palette(),
            'typography': self._create_typography_system(),
            'spacing': self._create_spacing_system(),
            'components': self._create_component_library(),
            'guidelines': self._create_design_guidelines()
        }

        # Save design system
        design_file = project_path / "src" / "design-system.json"
        with open(design_file, 'w') as f:
            json.dump(design_system, f, indent=2)

        # Create CSS variables file
        self._create_css_variables(project_path, design_system)

        # Create component styles
        self._create_component_styles(project_path)

        return {
            'status': 'completed',
            'design_system_created': True,
            'files_created': [str(design_file)]
        }

    def _create_color_palette(self) -> Dict[str, Any]:
        """Create a modern color palette"""
        return {
            'primary': {
                '50': '#e3f2fd',
                '100': '#bbdefb',
                '200': '#90caf9',
                '300': '#64b5f6',
                '400': '#42a5f5',
                '500': '#2196f3',
                '600': '#1e88e5',
                '700': '#1976d2',
                '800': '#1565c0',
                '900': '#0d47a1'
            },
            'secondary': {
                '50': '#fce4ec',
                '100': '#f8bbd9',
                '200': '#f48fb1',
                '300': '#ec407a',
                '400': '#e91e63',
                '500': '#e91e63',
                '600': '#d81b60',
                '700': '#c2185b',
                '800': '#ad1457',
                '900': '#880e4f'
            },
            'success': {
                '50': '#e8f5e8',
                '500': '#4caf50',
                '900': '#1b5e20'
            },
            'warning': {
                '50': '#fff3e0',
                '500': '#ff9800',
                '900': '#e65100'
            },
            'error': {
                '50': '#ffebee',
                '500': '#f44336',
                '900': '#b71c1c'
            },
            'gray': {
                '50': '#fafafa',
                '100': '#f5f5f5',
                '200': '#eeeeee',
                '300': '#e0e0e0',
                '400': '#bdbdbd',
                '500': '#9e9e9e',
                '600': '#757575',
                '700': '#616161',
                '800': '#424242',
                '900': '#212121'
            }
        }

    def _create_typography_system(self) -> Dict[str, Any]:
        """Create typography system"""
        return {
            'font_family': {
                'primary': 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
                'secondary': 'Georgia, serif',
                'monospace': 'Consolas, Monaco, "Courier New", monospace'
            },
            'font_size': {
                'xs': '0.75rem',
                'sm': '0.875rem',
                'base': '1rem',
                'lg': '1.125rem',
                'xl': '1.25rem',
                '2xl': '1.5rem',
                '3xl': '1.875rem',
                '4xl': '2.25rem',
                '5xl': '3rem'
            },
            'font_weight': {
                'thin': 100,
                'light': 300,
                'normal': 400,
                'medium': 500,
                'semibold': 600,
                'bold': 700,
                'extrabold': 800,
                'black': 900
            },
            'line_height': {
                'none': 1,
                'tight': 1.25,
                'snug': 1.375,
                'normal': 1.5,
                'relaxed': 1.625,
                'loose': 2
            }
        }

    def _create_spacing_system(self) -> Dict[str, Any]:
        """Create consistent spacing system"""
        return {
            'scale': [0, 4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128],
            'container': {
                'sm': '640px',
                'md': '768px',
                'lg': '1024px',
                'xl': '1280px',
                '2xl': '1536px'
            }
        }

    def _create_component_library(self) -> Dict[str, Any]:
        """Create component library specifications"""
        return {
            'button': {
                'variants': ['primary', 'secondary', 'outline', 'ghost'],
                'sizes': ['sm', 'md', 'lg'],
                'states': ['default', 'hover', 'active', 'disabled', 'loading']
            },
            'card': {
                'variants': ['default', 'elevated', 'outlined', 'filled'],
                'padding': ['none', 'sm', 'md', 'lg']
            },
            'input': {
                'variants': ['default', 'filled', 'outlined'],
                'states': ['default', 'focus', 'error', 'success', 'disabled']
            }
        }

    def _create_design_guidelines(self) -> Dict[str, Any]:
        """Create design guidelines"""
        return {
            'principles': self.design_principles,
            'best_practices': [
                'Use consistent spacing throughout the application',
                'Maintain proper contrast ratios for accessibility',
                'Provide clear feedback for user interactions',
                'Design for mobile-first approach',
                'Use progressive enhancement',
                'Optimize for performance and loading speed'
            ],
            'accessibility': {
                'color_contrast': 'Minimum 4.5:1 for normal text, 3:1 for large text',
                'keyboard_navigation': 'All interactive elements must be keyboard accessible',
                'screen_reader': 'Use semantic HTML and ARIA labels',
                'focus_management': 'Visible focus indicators for all interactive elements'
            }
        }

    def _create_css_variables(self, project_path: Path, design_system: Dict):
        """Create CSS custom properties for the design system"""

        css_content = '''
/* Design System CSS Variables */

:root {
  /* Colors */
'''
        # Add color variables
        for color_group, shades in design_system['colors'].items():
            for shade, value in shades.items():
                css_content += f'  --color-{color_group}-{shade}: {value};\n'

        css_content += '\n  /* Typography */\n'
        # Add typography variables
        for key, value in design_system['typography']['font_size'].items():
            css_content += f'  --font-size-{key}: {value};\n'

        for key, value in design_system['typography']['font_weight'].items():
            css_content += f'  --font-weight-{key}: {value};\n'

        for key, value in design_system['typography']['line_height'].items():
            css_content += f'  --line-height-{key}: {value};\n'

        css_content += '\n  /* Spacing */\n'
        # Add spacing variables
        for i, value in enumerate(design_system['spacing']['scale']):
            css_content += f'  --space-{i}: {value}px;\n'

        css_content += '''
  /* Common utility variables */
  --border-radius-sm: 0.25rem;
  --border-radius-md: 0.375rem;
  --border-radius-lg: 0.5rem;
  --border-radius-xl: 0.75rem;
  --border-radius-2xl: 1rem;
  --border-radius-full: 9999px;

  --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
  --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
  --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
  --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);

  --transition-fast: 150ms ease-in-out;
  --transition-normal: 300ms ease-in-out;
  --transition-slow: 500ms ease-in-out;
}

/* Dark mode variables */
@media (prefers-color-scheme: dark) {
  :root {
    --color-background: var(--color-gray-900);
    --color-surface: var(--color-gray-800);
    --color-text: var(--color-gray-50);
    --color-text-secondary: var(--color-gray-300);
  }
}
'''

        css_file = project_path / "src" / "styles" / "variables.css"
        css_file.parent.mkdir(exist_ok=True)
        with open(css_file, 'w') as f:
            f.write(css_content)

    def _create_component_styles(self, project_path: Path):
        """Create component-specific styles"""

        # Button styles
        button_css = '''
/* Button Component Styles */

.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: none;
  border-radius: var(--border-radius-md);
  font-family: var(--font-family-primary);
  font-size: var(--font-size-base);
  font-weight: var(--font-weight-medium);
  line-height: var(--line-height-none);
  text-decoration: none;
  cursor: pointer;
  transition: var(--transition-fast);
  user-select: none;
  white-space: nowrap;
  padding: var(--space-2) var(--space-4);
}

.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.btn--primary {
  background-color: var(--color-primary-500);
  color: white;
}

.btn--primary:hover:not(:disabled) {
  background-color: var(--color-primary-600);
}

.btn--secondary {
  background-color: var(--color-secondary-500);
  color: white;
}

.btn--secondary:hover:not(:disabled) {
  background-color: var(--color-secondary-600);
}

.btn--outline {
  background-color: transparent;
  color: var(--color-primary-500);
  border: 1px solid var(--color-primary-500);
}

.btn--outline:hover:not(:disabled) {
  background-color: var(--color-primary-500);
  color: white;
}

.btn--ghost {
  background-color: transparent;
  color: var(--color-primary-500);
}

.btn--ghost:hover:not(:disabled) {
  background-color: var(--color-gray-100);
}

.btn--sm {
  font-size: var(--font-size-sm);
  padding: var(--space-1) var(--space-3);
}

.btn--lg {
  font-size: var(--font-size-lg);
  padding: var(--space-3) var(--space-6);
}
'''
        with open(project_path / "src" / "styles" / "button.css", 'w') as f:
            f.write(button_css)

        # Card styles
        card_css = '''
/* Card Component Styles */

.card {
  background-color: var(--color-surface);
  border-radius: var(--border-radius-lg);
  box-shadow: var(--shadow-md);
  overflow: hidden;
  transition: var(--transition-normal);
}

.card:hover {
  box-shadow: var(--shadow-lg);
  transform: translateY(-2px);
}

.card__header {
  padding: var(--space-4) var(--space-6);
  border-bottom: 1px solid var(--color-gray-200);
}

.card__body {
  padding: var(--space-6);
}

.card__footer {
  padding: var(--space-4) var(--space-6);
  border-top: 1px solid var(--color-gray-200);
  background-color: var(--color-gray-50);
}

.card--elevated {
  box-shadow: var(--shadow-xl);
}

.card--outlined {
  border: 1px solid var(--color-gray-300);
  box-shadow: none;
}

.card--filled {
  background-color: var(--color-primary-50);
  border-color: var(--color-primary-200);
}
'''
        with open(project_path / "src" / "styles" / "card.css", 'w') as f:
            f.write(card_css)

    async def _implement_responsive_design(self, task) -> Dict[str, Any]:
        """Implement responsive design patterns"""

        responsive_css = '''
/* Responsive Design Utilities */

/* Responsive breakpoints */
@media (max-width: 640px) {
  .container {
    padding-left: var(--space-4);
    padding-right: var(--space-4);
  }

  .grid {
    grid-template-columns: 1fr;
  }

  .btn {
    width: 100%;
    justify-content: center;
  }
}

@media (min-width: 641px) and (max-width: 768px) {
  .container {
    max-width: 640px;
  }

  .grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (min-width: 769px) and (max-width: 1024px) {
  .container {
    max-width: 768px;
  }

  .grid {
    grid-template-columns: repeat(3, 1fr);
  }
}

@media (min-width: 1025px) {
  .container {
    max-width: 1024px;
  }

  .grid {
    grid-template-columns: repeat(4, 1fr);
  }
}

/* Mobile-first responsive utilities */
.hidden-mobile {
  display: none;
}

@media (min-width: 641px) {
  .hidden-mobile {
    display: block;
  }

  .hidden-desktop {
    display: none;
  }
}

/* Responsive typography */
.text-responsive {
  font-size: clamp(var(--font-size-base), 2.5vw, var(--font-size-xl));
}

/* Responsive spacing */
.p-responsive {
  padding: clamp(var(--space-2), 5vw, var(--space-8));
}

.m-responsive {
  margin: clamp(var(--space-2), 5vw, var(--space-8));
}
'''

        project_path = Path(self.agent_system.current_project.target_directory) / "frontend"
        responsive_file = project_path / "src" / "styles" / "responsive.css"
        with open(responsive_file, 'w') as f:
            f.write(responsive_css)

        return {
            'status': 'completed',
            'responsive_design_implemented': True,
            'files_created': [str(responsive_file)]
        }

    async def _implement_accessibility(self, task) -> Dict[str, Any]:
        """Implement accessibility features"""

        accessibility_css = '''
/* Accessibility Styles */

/* Focus indicators */
*:focus {
  outline: 2px solid var(--color-primary-500);
  outline-offset: 2px;
}

*:focus:not(:focus-visible) {
  outline: none;
}

*:focus-visible {
  outline: 2px solid var(--color-primary-500);
  outline-offset: 2px;
}

/* Skip links */
.skip-link {
  position: absolute;
  top: -40px;
  left: 6px;
  background: var(--color-primary-500);
  color: white;
  padding: 8px;
  text-decoration: none;
  border-radius: var(--border-radius-md);
  z-index: 1000;
}

.skip-link:focus {
  top: 6px;
}

/* High contrast mode */
@media (prefers-contrast: high) {
  :root {
    --color-primary-500: #0000ff;
    --color-text: #000000;
    --color-background: #ffffff;
  }

  .btn {
    border: 2px solid currentColor;
  }
}

/* Reduced motion */
@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}

/* Screen reader only */
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}

/* Screen reader only when focused */
.sr-only-focusable:focus {
  position: static;
  width: auto;
  height: auto;
  padding: inherit;
  margin: inherit;
  overflow: visible;
  clip: auto;
  white-space: normal;
}
'''

        project_path = Path(self.agent_system.current_project.target_directory) / "frontend"
        accessibility_file = project_path / "src" / "styles" / "accessibility.css"
        with open(accessibility_file, 'w') as f:
            f.write(accessibility_css)

        return {
            'status': 'completed',
            'accessibility_implemented': True,
            'files_created': [str(accessibility_file)]
        }

    async def _create_theme(self, task) -> Dict[str, Any]:
        """Create theme configuration"""

        theme_content = '''
// Theme Configuration
export const lightTheme = {
  colors: {
    primary: '#2196f3',
    secondary: '#e91e63',
    success: '#4caf50',
    warning: '#ff9800',
    error: '#f44336',
    background: '#ffffff',
    surface: '#f5f5f5',
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

export const darkTheme = {
  colors: {
    primary: '#64b5f6',
    secondary: '#f06292',
    success: '#81c784',
    warning: '#ffb74d',
    error: '#ef5350',
    background: '#121212',
    surface: '#1e1e1e',
    text: '#ffffff',
    textSecondary: '#b3b3b3'
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

export const getTheme = (mode = 'light') => {
  return mode === 'dark' ? darkTheme : lightTheme;
};
'''

        project_path = Path(self.agent_system.current_project.target_directory) / "frontend"
        theme_file = project_path / "src" / "theme.js"
        with open(theme_file, 'w') as f:
            f.write(theme_content)

        return {
            'status': 'completed',
            'theme_created': True,
            'files_created': [str(theme_file)]
        }

    async def _general_ui_task(self, task) -> Dict[str, Any]:
        """Handle general UI/UX tasks"""
        return {
            'status': 'completed',
            'message': f'UI/UX task completed: {task.description}'
        }