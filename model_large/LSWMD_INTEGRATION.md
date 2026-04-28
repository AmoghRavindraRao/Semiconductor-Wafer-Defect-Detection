# Wafer Defect Detection: LSWMD.pkl Integration

## Overview

The project has been successfully updated to use data from `LSWMD.pkl` instead of the `small_dataset` folder. The changes include:

- **Automatic data cache generation** from LSWMD.pkl
- **Enhanced `data_utils.py`** with LSWMD loading and preprocessing
- **Updated training/evaluation scripts** with optional LSWMD.pkl path
- **Data cache** created and saved in `data_cache/small_arrays.npz`

## Dataset Summary

**LSWMD.pkl Statistics:**
- Total samples: 811,457
- Labeled samples: 172,950
- Valid processed samples: 172,950

**Data Split:**
- **Training**: 48,920 samples
- **Validation**: 5,435 samples  
- **Test**: 118,595 samples

**Class Distribution:**
```
Center:     4,294 samples (2.5%)
Donut:        555 samples (0.3%)
Edge-Loc:   5,189 samples (3.0%)
Edge-Ring:  9,680 samples (5.6%)
Loc:        3,593 samples (2.1%)
Near-full:    149 samples (0.1%)
Random:       866 samples (0.5%)
Scratch:    1,193 samples (0.7%)
none:     147,431 samples (85.3%)
```

## Files Modified

### 1. `files/data_utils.py`
**Changes:**
- Added `load_lswmd_and_create_cache()` function to process LSWMD.pkl and create NPZ cache
- Added helper functions:
  - `_load_lswmd_pkl()`: Loads pickle with compatibility for older pandas versions
  - `_extract_label_from_array()`: Extracts labels from nested array format
  - `_process_wafer_map()`: Applies canonical preprocessing to wafer maps
- Enhanced `load_small_arrays()` to auto-generate cache if missing

**Key Features:**
- Handles pandas compatibility issues (old pickle format)
- Processes wafer maps to canonical 64×64 format
- Uses provided train/test split from LSWMD.pkl when available
- Falls back to random split if not available
- Reports comprehensive statistics during processing

### 2. `files/train_both.py`
**Changes:**
- Added `--lswmd_pkl` argument to specify LSWMD.pkl path
- Modified `load_small_arrays()` call to pass optional lswmd_pkl_path

**Usage:**
```bash
python train_both.py --lswmd_pkl ../LSWMD.pkl
```

### 3. `files/evaluate_both.py`
**Changes:**
- Added `--lswmd_pkl` argument
- Modified `load_small_arrays()` call to pass optional lswmd_pkl_path

**Usage:**
```bash
python evaluate_both.py --lswmd_pkl ../LSWMD.pkl
```

### 4. `files/create_data_cache.py` (New)
**Purpose:** Standalone script to generate data cache from LSWMD.pkl

**Usage:**
```bash
# Auto-detect LSWMD.pkl
python create_data_cache.py

# Or specify explicit path
python create_data_cache.py --lswmd_pkl path/to/LSWMD.pkl

# Options
python create_data_cache.py \
  --lswmd_pkl ../LSWMD.pkl \
  --output_dir data_cache \
  --val_split 0.1 \
  --test_split 0.1
```

## Quick Start

### Generate Data Cache (if needed)
```bash
cd files
python create_data_cache.py --lswmd_pkl ../LSWMD.pkl
```

### Train Models
```bash
cd files

# Using auto-generated cache
python train_both.py

# Or regenerate from LSWMD.pkl if cache missing
python train_both.py --lswmd_pkl ../LSWMD.pkl
```

### Evaluate Models
```bash
cd files
python evaluate_both.py
```

## Data Processing Pipeline

1. **Load LSWMD.pkl**: Extract DataFrame with 811,457 wafer maps
2. **Filter Labeled Samples**: Keep only samples with non-null failureType (172,950)
3. **Map Labels**: Convert failure types to class indices (0-8)
4. **Process Wafer Maps**: 
   - Clip to {0, 1, 2}
   - Crop to bounding box
   - Pad to square
   - Resize to 64×64 with nearest-neighbor interpolation
5. **Split Data**: Use provided train/test split from LSWMD.pkl
6. **Save as NPZ**: Store in `data_cache/small_arrays.npz`

## Class Mapping

```python
CLASS_TO_IDX = {
    "Center":    0,   "Donut":     1,   "Edge-Loc":  2,   "Edge-Ring": 3,
    "Loc":       4,   "Near-full": 5,   "Random":    6,   "Scratch":   7,
    "none":      8,
}
```

## Cached Data Files

### data_cache/small_arrays.npz
NPZ archive containing:
- `train_x`: (48920, 64, 64) uint8
- `train_y`: (48920,) int64
- `val_x`: (5435, 64, 64) uint8
- `val_y`: (5435,) int64
- `test_x`: (118595, 64, 64) uint8
- `test_y`: (118595,) int64

## Backward Compatibility

The system maintains backward compatibility:
- If `data_cache/small_arrays.npz` exists, it uses the cached version
- If cache is missing but LSWMD.pkl is available, it generates cache automatically
- Falls back to `small_dataset` folder if both are missing (original behavior)

## Important Notes

1. **First Run**: Initial cache creation takes several minutes due to LSWMD.pkl processing
2. **Data Split**: Uses the provided "Training" vs "Test" labels from LSWMD.pkl
3. **Class Imbalance**: The "none" class represents ~85% of the data
4. **Validation Split**: Extracted 10% of training data for validation

## Troubleshooting

### "LSWMD.pkl not found"
```bash
# Ensure LSWMD.pkl is in the correct location
# Check these paths:
- C:\MIX\ASU\SEM_4\DSE 570\project_leakage\data\LSWMD.pkl
- ../LSWMD.pkl (relative to files/ directory)
- ../../LSWMD.pkl
```

### "Cache creation failed"
```bash
# Regenerate cache with verbose output
python create_data_cache.py --lswmd_pkl ../LSWMD.pkl
```

### Import errors
Ensure pandas and other dependencies are installed:
```bash
pip install torch torchvision pillow matplotlib scikit-learn seaborn numpy pandas opencv-python
```

## Summary of Changes

✓ LSWMD.pkl now serves as the primary data source  
✓ Automatic cache generation on first run  
✓ 172,950 labeled wafers processed and cached  
✓ Data split: 48,920 train / 5,435 val / 118,595 test  
✓ Full backward compatibility maintained  
✓ Comprehensive documentation provided  

