#!/usr/bin/env python
"""
create_data_cache.py

Generates data_cache/small_arrays.npz from LSWMD.pkl.

Usage:
    python create_data_cache.py --lswmd_pkl path/to/LSWMD.pkl
    python create_data_cache.py  # uses default location
"""
import argparse
from pathlib import Path
import sys

# Add files directory to path to import data_utils
sys.path.insert(0, str(Path(__file__).parent))

from data_utils import load_lswmd_and_create_cache

def main():
    parser = argparse.ArgumentParser(
        description="Create data cache from LSWMD.pkl",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--lswmd_pkl",
        type=Path,
        default=None,
        help="Path to LSWMD.pkl file. If not provided, searches common locations."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data_cache"),
        help="Output directory for NPZ files"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Validation split ratio"
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.1,
        help="Test split ratio"
    )
    args = parser.parse_args()

    # Find LSWMD.pkl if not provided
    lswmd_pkl = args.lswmd_pkl
    if not lswmd_pkl:
        search_paths = [
            Path("LSWMD.pkl"),
            Path("../LSWMD.pkl"),
            Path("../../LSWMD.pkl"),
            Path(__file__).parent / "LSWMD.pkl",
            Path(__file__).parent.parent / "LSWMD.pkl",
        ]
        for p in search_paths:
            if p.exists():
                lswmd_pkl = p
                print(f"Found LSWMD.pkl at: {p.resolve()}")
                break
    
    if not lswmd_pkl or not Path(lswmd_pkl).exists():
        print(f"Error: LSWMD.pkl not found at {lswmd_pkl}")
        print(f"Please provide --lswmd_pkl with the correct path")
        print(f"\nSearched locations:")
        for p in [Path("LSWMD.pkl"), Path("../LSWMD.pkl"), Path("../../LSWMD.pkl")]:
            print(f"  - {p.resolve()}")
        return 1

    print(f"Creating cache from: {Path(lswmd_pkl).resolve()}")
    print(f"Output directory: {args.output_dir.resolve()}")
    print()

    try:
        result = load_lswmd_and_create_cache(
            str(lswmd_pkl),
            str(args.output_dir),
            val_split=args.val_split,
            test_split=args.test_split,
        )
        print(f"\n✓ Successfully created data cache!")
        print(f"  Train: {len(result['train_x'])} samples")
        print(f"  Val:   {len(result['val_x'])} samples")
        print(f"  Test:  {len(result['test_x'])} samples")
        return 0
    except Exception as e:
        print(f"✗ Error creating cache: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
