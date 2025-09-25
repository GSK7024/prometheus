#!/usr/bin/env python3
"""
Simple launcher for Enhanced Prometheus AI
Makes it easy to run the system with different goals
"""

import sys
import argparse
from pathlib import Path

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Enhanced Prometheus AI - Ultimate AI Coding System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prometheus_launcher.py --goal "Create a web app" --project my_app
  python prometheus_launcher.py --goal "Design a rocket system" --project rocket_design
  python prometheus_launcher.py --demo mars_mission
  python prometheus_launcher.py --test
        """
    )

    parser.add_argument("--goal", "-g", help="Project goal/description")
    parser.add_argument("--project", "-p", help="Project name")
    parser.add_argument("--agent", "-a", help="Main agent file", default="main.py")
    parser.add_argument("--demo", "-d", help="Run demo (mars_mission)")
    parser.add_argument("--test", "-t", action="store_true", help="Run system tests")
    parser.add_argument("--strategy", "-s", default="cognitive",
                       help="Development strategy (tdd, agile, devops, cognitive)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    return parser.parse_args()

def run_demo(demo_type):
    """Run a demonstration"""
    if demo_type == "mars_mission":
        print("🚀 Running Mars Mission Control Demo...")
        try:
            exec(open("mars_mission_final_demo.py").read())
        except FileNotFoundError:
            print("❌ Demo file not found. Please ensure mars_mission_final_demo.py exists.")
    else:
        print(f"❌ Unknown demo type: {demo_type}")

def run_test():
    """Run system tests"""
    print("🧪 Running system tests...")
    try:
        exec(open("test_system.py").read())
    except FileNotFoundError:
        print("❌ Test file not found. Please ensure test_system.py exists.")

def run_main_system(args):
    """Run the main Prometheus AI system"""
    print("🚀 Starting Enhanced Prometheus AI...")
    print(f"Goal: {args.goal}")
    print(f"Project: {args.project}")
    print(f"Agent: {args.agent}")
    print(f"Strategy: {args.strategy}")
    print()

    try:
        from prometheus import UltraCognitiveForgeOrchestrator

        # Create the orchestrator
        orchestrator = UltraCognitiveForgeOrchestrator(
            goal=args.goal,
            project_name=args.project,
            agent_filename=args.agent,
            dev_strategy=args.strategy
        )

        print("✅ Orchestrator created successfully!")
        print("🔧 System is ready to process your request...")
        print()
        print("Note: Full execution requires API keys in .env file")
        print("For a full demo, try: python prometheus_launcher.py --demo mars_mission")

    except ImportError as e:
        print(f"❌ Failed to import Prometheus: {e}")
        print("Please run: python setup.py")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Main launcher function"""
    print("🚀 Enhanced Prometheus AI Launcher")
    print("=" * 50)

    args = parse_arguments()

    if args.test:
        run_test()
    elif args.demo:
        run_demo(args.demo)
    elif args.goal and args.project:
        run_main_system(args)
    else:
        print("❌ Please provide either --goal and --project, --demo, or --test")
        print("\nFor help: python prometheus_launcher.py --help")

if __name__ == "__main__":
    main()