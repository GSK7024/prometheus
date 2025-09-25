#!/usr/bin/env python3
"""
Main entry point for the P2P Chat Application.
This is a simple, local, command-line peer-to-peer chat application using Python.
"""

import asyncio
import logging
from config import Config
from p2p_core import P2PChatCore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main application entry point."""
    logger.info("Starting P2P Chat Application...")

    try:
        # Load configuration
        config = Config()
        logger.info("Configuration loaded successfully")

        # Initialize P2P core
        chat_core = P2PChatCore(config)
        await chat_core.initialize()

        logger.info("P2P Chat Application initialized successfully")
        logger.info("Use Ctrl+C to exit the application")

        # Keep the application running
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("Application shutdown requested by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())