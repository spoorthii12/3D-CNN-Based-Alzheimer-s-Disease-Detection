import os
import nibabel as nib
import numpy as np
from skimage.transform import resize
from tqdm import tqdm

# Where your raw OASIS folders live:
INPUT_DIRS = ['OAS2_RAW_PART1', 'OAS2_RAW_PART2']

# Where to dump 128×128×128 .npy volumes
OUTPUT_DIR = 'data/processed'
TARGET_SHAPE = (128, 128, 128)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def preprocess_scan(subject_dir):
    """
    - Looks for mpr-1.img/.hdr or falls back to mpr-1.nifti.img/.hdr
    - Loads, squeezes singleton dimensions, normalizes to [0,1],
      resizes to TARGET_SHAPE, returns float32 volume.
    """
    raw = os.path.join(subject_dir, 'RAW')
    # Try Analyze format
    img_path = os.path.join(raw, 'mpr-1.img')
    hdr_path = os.path.join(raw, 'mpr-1.hdr')
    # Fallback to NIfTI format
    if not (os.path.exists(img_path) and os.path.exists(hdr_path)):
        img_path = os.path.join(raw, 'mpr-1.nifti.img')
        hdr_path = os.path.join(raw, 'mpr-1.nifti.hdr')
    if not (os.path.exists(img_path) and os.path.exists(hdr_path)):
        print(f"Missing mpr-1 in {raw}, skipping…")
        return None

    try:
        img = nib.load(img_path)
        data = img.get_fdata()
    except Exception as e:
        print(f"❌ Couldn’t load {img_path}: {e}")
        return None

    # Remove any trailing singleton dims, e.g., (X,Y,Z,1) → (X,Y,Z)
    data = np.squeeze(data)

    # Min-max normalization to [0, 1]
    data = (data - data.min()) / (data.max() - data.min())

    # Resize to (128,128,128)
    data = resize(data, TARGET_SHAPE, mode='constant', anti_aliasing=True)

    return data.astype(np.float32)

if __name__ == '__main__':
    for root in INPUT_DIRS:
        print(f"\n>> Processing {root} …")
        for subj in tqdm(os.listdir(root)):
            subj_dir = os.path.join(root, subj)
            if not os.path.isdir(subj_dir):
                continue
            vol = preprocess_scan(subj_dir)
            if vol is None:
                continue
            # Use MRI ID format matching tabular data
            mri_id = subj  # e.g., 'OAS2_0079_MR2'
            out_path = os.path.join(OUTPUT_DIR, f"{mri_id}.npy")
            np.save(out_path, vol)
    print("\n✅ Done.")