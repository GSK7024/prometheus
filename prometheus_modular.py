#!/usr/bin/env python3
"""
Prometheus AI Orchestrator - Modularized Version
A sophisticated AI system that has been split into multiple modules for better maintainability
"""

import asyncio
from prometheus_modules.orchestrator import main

if __name__ == "__main__":
    print("🚀 Starting Prometheus AI Orchestrator (Modularized Version)")
    print("📦 Components:")
    print("  • prometheus_modules.config - Configuration management")
    print("  • prometheus_modules.utils - Utility functions and logging")
    print("  • prometheus_modules.models - Data models and enums")
    print("  • prometheus_modules.core_ai - Quantum cognitive core and memory")
    print("  • prometheus_modules.coverage - Coverage scoring utilities")
    print("  • prometheus_modules.orchestrator - Main orchestrator logic")
    print("\n🎯 Enter your project goal when prompted...")
    print()

    asyncio.run(main())