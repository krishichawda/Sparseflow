"""
SparseFlow command-line interface
"""

import sys
import argparse
from pathlib import Path

# Add examples to path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(
        description='SparseFlow: Adaptive Sparse Model Offloading'
    )
    parser.add_argument(
        'command',
        choices=['demo', 'version', 'info'],
        help='Command to execute'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='facebook/opt-1.3b',
        help='Model name (default: facebook/opt-1.3b)'
    )
    parser.add_argument(
        '--cache-ratio',
        type=float,
        default=0.3,
        help='GPU cache ratio 0.0-1.0 (default: 0.3)'
    )
    
    args = parser.parse_args()
    
    if args.command == 'version':
        from sparseflow import __version__
        print(f"SparseFlow version {__version__}")
    
    elif args.command == 'info':
        print("SparseFlow: Adaptive Sparse Model Offloading")
        print("\nCommands:")
        print("  demo        Run demo generation")
        print("  version     Show version")
        print("  info        Show this information")
        print("\nFor more information, see: https://github.com/yourusername/sparseflow")
    
    elif args.command == 'demo':
        print("Running SparseFlow demo...")
        from sparseflow.examples import demo_generation
        demo_generation.main()


if __name__ == '__main__':
    main()

