#!/usr/bin/env python3
"""Preprocess HCP parcellated data into expected format for Brain-JEPA"""
import os
import json
import glob
import torch
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

try:
    import boto3
    from io import BytesIO
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False
    print("Warning: boto3 not available. S3 upload disabled.")

def load_label_mapping(json_path):
    """Load subject ID to label mapping"""
    with open(json_path, 'r') as f:
        mapping = json.load(f)
    # Keep as string keys (matching meta['sub'] format) and convert values to int
    return {str(k): int(v) for k, v in mapping.items()}

def process_pt_file(filepath, s3_client=None, s3_bucket=None, s3_key=None):
    """Load and process a single .pt file from local path or S3"""
    if s3_client and s3_bucket and s3_key:
        # Load from S3
        response = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
        data = torch.load(BytesIO(response['Body'].read()), map_location='cpu')
    else:
        # Load from local file
        data = torch.load(filepath, map_location='cpu')
    
    # Extract subject ID from meta (it's stored as a string)
    subject_id = str(data['meta']['sub'])
    
    # Extract bold signal [n_frames, 450]
    bold = data['bold']  # Shape: [n_frames, 450]
    
    # Transpose to [450, n_frames] to match expected format (ROIs x timepoints)
    bold = bold.T  # Now shape: [450, n_frames]
    
    # Pad or interpolate to seq_length=490
    n_frames = bold.shape[1]
    seq_length = 490
    
    if n_frames < seq_length:
        # Pad with last value
        padding = bold[:, -1:].repeat(1, seq_length - n_frames)
        bold = torch.cat([bold, padding], dim=1)
    elif n_frames > seq_length:
        # Downsample/interpolate to 490
        indices = torch.linspace(0, n_frames - 1, seq_length).long()
        bold = bold[:, indices]
    
    return subject_id, bold.float()

def list_s3_files(s3_bucket, s3_prefix, max_files=None):
    """List all .pt files in S3 prefix"""
    if not HAS_BOTO3:
        raise ImportError("boto3 required for S3 operations")
    
    s3 = boto3.client('s3')
    pt_files = []
    
    print(f"Listing .pt files in s3://{s3_bucket}/{s3_prefix}/...")
    paginator = s3.get_paginator('list_objects_v2')
    
    for page in paginator.paginate(Bucket=s3_bucket, Prefix=s3_prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                if obj['Key'].endswith('.pt') and 'hca450_' not in obj['Key']:
                    pt_files.append({
                        'key': obj['Key'],
                        'size': obj['Size']
                    })
                    if max_files and len(pt_files) >= max_files:
                        break
        if max_files and len(pt_files) >= max_files:
            break
    
    print(f"Found {len(pt_files)} .pt files")
    return pt_files

def preprocess_hcp_data_from_s3(
    s3_bucket='medarc',
    s3_prefix='fmri-fm/datasets/hcp-parc-v2',
    s3_output_prefix=None,  # If None, use same as input
    label_map_path='hcp_sex_target_id_map.json',
    output_dir=None,  # Local output dir (optional, for debugging)
    max_files=None,  # Limit number of files for testing
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42
):
    """Preprocess HCP data directly from S3, write results back to S3"""
    if not HAS_BOTO3:
        raise ImportError("boto3 required for S3 operations")
    
    s3 = boto3.client('s3')
    s3_output_prefix = s3_output_prefix or s3_prefix
    
    print(f"\n{'='*60}")
    print("PREPROCESSING FROM S3")
    print(f"{'='*60}")
    print(f"Input:  s3://{s3_bucket}/{s3_prefix}/")
    print(f"Output: s3://{s3_bucket}/{s3_output_prefix}/")
    if max_files:
        print(f"Limit:  {max_files} files (testing mode)")
    print(f"{'='*60}\n")
    
    # Load label mapping
    print(f"Loading label mapping from {label_map_path}...")
    label_mapping = load_label_mapping(label_map_path)
    print(f"Loaded {len(label_mapping)} subject labels")
    
    # List S3 files
    s3_files = list_s3_files(s3_bucket, s3_prefix, max_files=max_files)
    
    if len(s3_files) == 0:
        raise ValueError("No .pt files found in S3!")
    
    # Process files and group by subject
    print("\nProcessing files from S3 and grouping by subject...")
    subject_data = {}
    missing_labels = []
    
    for i, file_info in enumerate(tqdm(s3_files, desc="Processing")):
        s3_key = file_info['key']
        
        try:
            subject_id, bold_tensor = process_pt_file(
                filepath=None,
                s3_client=s3,
                s3_bucket=s3_bucket,
                s3_key=s3_key
            )
            
            # Get label from mapping
            if subject_id in label_mapping:
                label = label_mapping[subject_id]
                
                if subject_id not in subject_data:
                    subject_data[subject_id] = {'features': [], 'label': label}
                
                subject_data[subject_id]['features'].append(bold_tensor)
            else:
                missing_labels.append((subject_id, s3_key))
        except Exception as e:
            print(f"\n  Error processing {s3_key}: {e}")
            continue
    
    print(f"\nSuccessfully processed {len(s3_files)} files")
    print(f"  Unique subjects: {len(subject_data)}")
    print(f"  Total samples: {sum(len(v['features']) for v in subject_data.values())}")
    
    if missing_labels:
        print(f"\nWarning: {len(missing_labels)} files had missing labels:")
        for sub_id, key in missing_labels[:10]:
            print(f"  Subject {sub_id}: {key}")
        if len(missing_labels) > 10:
            print(f"  ... and {len(missing_labels) - 10} more")
    
    if len(subject_data) == 0:
        raise ValueError("No valid files processed!")
    
    # Prepare subject-level data for splitting
    subjects = list(subject_data.keys())
    subject_labels = [subject_data[sub]['label'] for sub in subjects]
    
    print(f"\nSubject-level statistics:")
    print(f"  Total subjects: {len(subjects)}")
    print(f"  Label distribution: {torch.bincount(torch.tensor(subject_labels)).tolist()}")
    
    # Split subjects (not individual samples) into train/val/test
    print(f"\nSplitting subjects (train={train_ratio:.1%}, val={val_ratio:.1%}, test={test_ratio:.1%})...")
    
    train_subjects, temp_subjects, train_subject_labels, temp_subject_labels = train_test_split(
        subjects, subject_labels,
        test_size=(val_ratio + test_ratio),
        random_state=seed,
        stratify=subject_labels
    )
    
    val_size = val_ratio / (val_ratio + test_ratio)
    val_subjects, test_subjects, val_subject_labels, test_subject_labels = train_test_split(
        temp_subjects, temp_subject_labels,
        test_size=(1 - val_size),
        random_state=seed,
        stratify=temp_subject_labels
    )
    
    # Collect all samples from each subject group
    train_features = []
    train_labels = []
    for sub_id in train_subjects:
        train_features.extend(subject_data[sub_id]['features'])
        train_labels.extend([subject_data[sub_id]['label']] * len(subject_data[sub_id]['features']))
    
    val_features = []
    val_labels = []
    for sub_id in val_subjects:
        val_features.extend(subject_data[sub_id]['features'])
        val_labels.extend([subject_data[sub_id]['label']] * len(subject_data[sub_id]['features']))
    
    test_features = []
    test_labels = []
    for sub_id in test_subjects:
        test_features.extend(subject_data[sub_id]['features'])
        test_labels.extend([subject_data[sub_id]['label']] * len(subject_data[sub_id]['features']))
    
    print(f"\nFinal split statistics:")
    print(f"  Train: {len(train_subjects)} subjects → {len(train_features)} samples")
    print(f"  Val:   {len(val_subjects)} subjects → {len(val_features)} samples")
    print(f"  Test:  {len(test_subjects)} subjects → {len(test_features)} samples")
    
    # Save locally first (for debugging/verification)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nSaving processed data locally to {output_dir}/...")
        torch.save(train_features, os.path.join(output_dir, 'hca450_train_x.pt'))
        torch.save(train_labels, os.path.join(output_dir, 'hca450_train_y.pt'))
        torch.save(val_features, os.path.join(output_dir, 'hca450_valid_x.pt'))
        torch.save(val_labels, os.path.join(output_dir, 'hca450_valid_y.pt'))
        torch.save(test_features, os.path.join(output_dir, 'hca450_test_x.pt'))
        torch.save(test_labels, os.path.join(output_dir, 'hca450_test_y.pt'))
    
    # Upload to S3
    print(f"\nUploading processed files to S3...")
    files_to_upload = [
        ('hca450_train_x.pt', train_features),
        ('hca450_train_y.pt', train_labels),
        ('hca450_valid_x.pt', val_features),
        ('hca450_valid_y.pt', val_labels),
        ('hca450_test_x.pt', test_features),
        ('hca450_test_y.pt', test_labels),
    ]
    
    uploaded = 0
    for filename, data in tqdm(files_to_upload, desc="Uploading", unit="file"):
        s3_key = f"{s3_output_prefix}/{filename}"
        
        # Save to BytesIO buffer
        buffer = BytesIO()
        torch.save(data, buffer)
        buffer.seek(0)
        
        try:
            s3.upload_fileobj(buffer, s3_bucket, s3_key)
            uploaded += 1
            print(f"  ✓ Uploaded {filename} → s3://{s3_bucket}/{s3_key}")
        except Exception as e:
            print(f"  ✗ Error uploading {filename}: {e}")
    
    print(f"\n{'='*60}")
    print("PREPROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Uploaded: {uploaded}/{len(files_to_upload)} files")
    print(f"Location: s3://{s3_bucket}/{s3_output_prefix}/")
    if output_dir:
        print(f"Local copy: {output_dir}/")
    print(f"{'='*60}")
    
    return {
        'train_features': train_features,
        'train_labels': train_labels,
        'val_features': val_features,
        'val_labels': val_labels,
        'test_features': test_features,
        'test_labels': test_labels,
        's3_bucket': s3_bucket,
        's3_prefix': s3_output_prefix
    }

def preprocess_hcp_data(
    data_dir='data',
    label_map_path='hcp_sex_target_id_map.json',
    output_dir='brain-jepa-dataset',
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42
):
    """Preprocess HCP data into expected format"""
    
    # Validate split ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Split ratios must sum to 1.0"
    
    print(f"Loading label mapping from {label_map_path}...")
    label_mapping = load_label_mapping(label_map_path)
    print(f"Loaded {len(label_mapping)} subject labels")
    
    # Find all .pt files
    print(f"\nScanning for .pt files in {data_dir}...")
    pt_files = []
    
    # Try multiple patterns to find files
    patterns = [
        f'{data_dir}/hcp-parc_*/*.pt',  # Direct in hcp-parc_* directories
        f'{data_dir}/hcp-parc_*/*/*.pt',  # In subdirectories
        f'{data_dir}/**/*.pt',  # Recursive
    ]
    
    for pattern in patterns:
        found = glob.glob(pattern, recursive=True)
        pt_files.extend(found)
    
    # Remove duplicates and exclude already-processed files
    pt_files = list(set(pt_files))
    pt_files = [f for f in pt_files if 'hca450_' not in f]  # Exclude processed files
    pt_files.sort()
    
    print(f"Found {len(pt_files)} .pt files")
    
    # Process files and group by subject
    print("\nProcessing files and grouping by subject...")
    subject_data = {}  # {subject_id: {'features': [...], 'label': int}}
    missing_labels = []
    
    for i, filepath in enumerate(pt_files):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(pt_files)} files...")
        
        try:
            subject_id, bold_tensor = process_pt_file(filepath)
            
            # Get label from mapping
            if subject_id in label_mapping:
                label = label_mapping[subject_id]
                
                # Group by subject
                if subject_id not in subject_data:
                    subject_data[subject_id] = {'features': [], 'label': label}
                
                subject_data[subject_id]['features'].append(bold_tensor)
            else:
                missing_labels.append((subject_id, filepath))
        except Exception as e:
            print(f"  Error processing {filepath}: {e}")
            continue
    
    print(f"\nSuccessfully processed {len(pt_files)} files")
    print(f"  Unique subjects: {len(subject_data)}")
    print(f"  Total samples: {sum(len(v['features']) for v in subject_data.values())}")
    
    if missing_labels:
        print(f"\nWarning: {len(missing_labels)} files had missing labels:")
        for sub_id, fpath in missing_labels[:10]:  # Show first 10
            print(f"  Subject {sub_id}: {fpath}")
        if len(missing_labels) > 10:
            print(f"  ... and {len(missing_labels) - 10} more")
    
    if len(subject_data) == 0:
        raise ValueError("No valid files processed!")
    
    # Prepare subject-level data for splitting
    subjects = list(subject_data.keys())
    subject_labels = [subject_data[sub]['label'] for sub in subjects]
    
    print(f"\nSubject-level statistics:")
    print(f"  Total subjects: {len(subjects)}")
    print(f"  Label distribution: {torch.bincount(torch.tensor(subject_labels)).tolist()}")
    
    # Split subjects (not individual samples) into train/val/test
    print(f"\nSplitting subjects (train={train_ratio:.1%}, val={val_ratio:.1%}, test={test_ratio:.1%})...")
    
    # First split: train vs (val+test)
    train_subjects, temp_subjects, train_subject_labels, temp_subject_labels = train_test_split(
        subjects, subject_labels,
        test_size=(val_ratio + test_ratio),
        random_state=seed,
        stratify=subject_labels
    )
    
    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_subjects, test_subjects, val_subject_labels, test_subject_labels = train_test_split(
        temp_subjects, temp_subject_labels,
        test_size=(1 - val_size),
        random_state=seed,
        stratify=temp_subject_labels
    )
    
    # Collect all samples from each subject group
    train_features = []
    train_labels = []
    for sub_id in train_subjects:
        train_features.extend(subject_data[sub_id]['features'])
        train_labels.extend([subject_data[sub_id]['label']] * len(subject_data[sub_id]['features']))
    
    val_features = []
    val_labels = []
    for sub_id in val_subjects:
        val_features.extend(subject_data[sub_id]['features'])
        val_labels.extend([subject_data[sub_id]['label']] * len(subject_data[sub_id]['features']))
    
    test_features = []
    test_labels = []
    for sub_id in test_subjects:
        test_features.extend(subject_data[sub_id]['features'])
        test_labels.extend([subject_data[sub_id]['label']] * len(subject_data[sub_id]['features']))
    
    print(f"\nFinal split statistics:")
    print(f"  Train: {len(train_subjects)} subjects → {len(train_features)} samples")
    print(f"  Val:   {len(val_subjects)} subjects → {len(val_features)} samples")
    print(f"  Test:  {len(test_subjects)} subjects → {len(test_features)} samples")
    print(f"\nSample-level label distribution:")
    print(f"  Train: {torch.bincount(torch.tensor(train_labels)).tolist()}")
    print(f"  Val:   {torch.bincount(torch.tensor(val_labels)).tolist()}")
    print(f"  Test:  {torch.bincount(torch.tensor(test_labels)).tolist()}")
    
    # Save files
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving processed data to {output_dir}/...")
    
    # Save as lists (as expected by the dataset loader)
    # Labels should be a list of integers, not a list of lists
    torch.save(train_features, os.path.join(output_dir, 'hca450_train_x.pt'))
    torch.save(train_labels, os.path.join(output_dir, 'hca450_train_y.pt'))
    
    torch.save(val_features, os.path.join(output_dir, 'hca450_valid_x.pt'))
    torch.save(val_labels, os.path.join(output_dir, 'hca450_valid_y.pt'))
    
    torch.save(test_features, os.path.join(output_dir, 'hca450_test_x.pt'))
    torch.save(test_labels, os.path.join(output_dir, 'hca450_test_y.pt'))
    
    print("✓ Preprocessing complete!")
    print(f"\nFiles saved:")
    print(f"  {output_dir}/hca450_train_x.pt ({len(train_features)} samples)")
    print(f"  {output_dir}/hca450_train_y.pt")
    print(f"  {output_dir}/hca450_valid_x.pt ({len(val_features)} samples)")
    print(f"  {output_dir}/hca450_valid_y.pt")
    print(f"  {output_dir}/hca450_test_x.pt ({len(test_features)} samples)")
    print(f"  {output_dir}/hca450_test_y.pt")
    
    return {
        'train_features': train_features,
        'train_labels': train_labels,
        'val_features': val_features,
        'val_labels': val_labels,
        'test_features': test_features,
        'test_labels': test_labels,
        'output_dir': output_dir
    }

def upload_to_s3(local_dir, s3_bucket='medarc', s3_prefix='fmri-fm/datasets/hcp-parc-v2', overwrite=False):
    """Upload processed files to S3"""
    if not HAS_BOTO3:
        raise ImportError("boto3 required for S3 upload. Install with: pip install boto3")
    
    s3 = boto3.client('s3')
    
    files_to_upload = [
        'hca450_train_x.pt',
        'hca450_train_y.pt',
        'hca450_valid_x.pt',
        'hca450_valid_y.pt',
        'hca450_test_x.pt',
        'hca450_test_y.pt'
    ]
    
    print(f"\n{'='*60}")
    print("UPLOADING TO S3")
    print(f"{'='*60}")
    print(f"Bucket: {s3_bucket}")
    print(f"Prefix: {s3_prefix}")
    print(f"{'='*60}\n")
    
    uploaded = 0
    skipped = 0
    
    for filename in tqdm(files_to_upload, desc="Uploading", unit="file"):
        local_path = os.path.join(local_dir, filename)
        
        if not os.path.exists(local_path):
            print(f"Warning: {local_path} not found, skipping...")
            continue
        
        s3_key = f"{s3_prefix}/{filename}"
        
        # Check if file exists on S3
        if not overwrite:
            try:
                s3.head_object(Bucket=s3_bucket, Key=s3_key)
                print(f"  Skipping {filename} (already exists on S3)")
                skipped += 1
                continue
            except s3.exceptions.ClientError:
                # File doesn't exist, proceed with upload
                pass
        
        try:
            file_size = os.path.getsize(local_path)
            print(f"  Uploading {filename} ({file_size / 1024 / 1024:.2f} MB)...")
            s3.upload_file(local_path, s3_bucket, s3_key)
            uploaded += 1
            print(f"  ✓ Uploaded to s3://{s3_bucket}/{s3_key}")
        except Exception as e:
            print(f"  ✗ Error uploading {filename}: {e}")
    
    print(f"\n{'='*60}")
    print("UPLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"Uploaded: {uploaded} files")
    print(f"Skipped:  {skipped} files")
    print(f"Location: s3://{s3_bucket}/{s3_prefix}/")
    print(f"{'='*60}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess HCP parcellated data')
    
    # Mode selection
    parser.add_argument('--s3_input', action='store_true',
                        help='Process data directly from S3 (instead of local directory)')
    
    # Local mode arguments
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing hcp-parc_* subdirectories (local mode)')
    
    # S3 mode arguments
    parser.add_argument('--s3_bucket', type=str, default='medarc',
                        help='S3 bucket name')
    parser.add_argument('--s3_prefix', type=str, default='fmri-fm/datasets/hcp-parc-v2',
                        help='S3 prefix/path')
    parser.add_argument('--s3_output_prefix', type=str, default=None,
                        help='S3 output prefix (default: same as input prefix)')
    parser.add_argument('--max_files', type=int, default=None,
                        help='Limit number of files to process (for testing)')
    
    # Common arguments
    parser.add_argument('--label_map', type=str, default='hcp_sex_target_id_map.json',
                        help='Path to label mapping JSON file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Local output directory (optional, for debugging)')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Test set ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for splitting')
    
    # Legacy upload flag (for local mode)
    parser.add_argument('--upload_s3', action='store_true',
                        help='Upload processed files to S3 (local mode only)')
    parser.add_argument('--s3_overwrite', action='store_true',
                        help='Overwrite existing files on S3')
    
    args = parser.parse_args()
    
    if args.s3_input:
        # S3 mode: process from S3 → write to S3
        preprocess_hcp_data_from_s3(
            s3_bucket=args.s3_bucket,
            s3_prefix=args.s3_prefix,
            s3_output_prefix=args.s3_output_prefix,
            label_map_path=args.label_map,
            output_dir=args.output_dir,
            max_files=args.max_files,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed
        )
    else:
        # Local mode: process from local → save locally (optionally upload to S3)
        result = preprocess_hcp_data(
            data_dir=args.data_dir,
            label_map_path=args.label_map,
            output_dir=args.output_dir or 'brain-jepa-dataset',
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed
        )
        
        if args.upload_s3:
            upload_to_s3(
                local_dir=result['output_dir'],
                s3_bucket=args.s3_bucket,
                s3_prefix=args.s3_prefix,
                overwrite=args.s3_overwrite
            )