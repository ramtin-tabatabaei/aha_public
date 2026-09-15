"""Combine the camera frames for exactly one selected task."""
import argparse
from pathlib import Path
from . import waypoints_combine_images as grids

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    args = parser.parse_args()
    grids.combine_folder(args.task, include_gripper=True)
    target = Path(grids.OUTPUT_FOLDER) / f'{args.task}_ALL_WAYPOINTS_COMBINED.png'
    if not target.exists():
        parser.error(f'No grid was generated for {args.task}')

if __name__ == '__main__':
    main()
