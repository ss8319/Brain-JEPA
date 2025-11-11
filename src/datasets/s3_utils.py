"""Utilities for S3 operations in dataset loading"""
import os
import re
import datetime
from pathlib import Path

try:
    import boto3
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False


def parse_s3_path(s3_path):
    """Parse s3://bucket/prefix/path into bucket and key"""
    if not s3_path.startswith('s3://'):
        return None, None
    
    match = re.match(r's3://([^/]+)/(.*)', s3_path)
    if match:
        bucket = match.group(1)
        prefix = match.group(2).rstrip('/')
        return bucket, prefix
    return None, None


def ensure_s3_files_cached(s3_path, cache_dir='~/.cache/brain-jepa', force_download=False):
    """
    Download processed dataset files from S3 to local cache if needed.
    
    Args:
        s3_path: S3 path like 's3://bucket/prefix' containing hca450_*_x.pt and hca450_*_y.pt files
        cache_dir: Local cache directory
        force_download: Force re-download even if files exist
    
    Returns:
        Local directory path containing cached files
    """
    if not HAS_BOTO3:
        raise ImportError("boto3 required for S3 operations. Install with: pip install boto3")
    
    bucket, prefix = parse_s3_path(s3_path)
    if not bucket or not prefix:
        raise ValueError(f"Invalid S3 path: {s3_path}")
    
    # Expand cache directory
    cache_dir = os.path.expanduser(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    
    s3 = boto3.client('s3')
    
    # Files we need to download
    files_to_download = [
        'hca450_train_x.pt',
        'hca450_train_y.pt',
        'hca450_valid_x.pt',
        'hca450_valid_y.pt',
        'hca450_test_x.pt',
        'hca450_test_y.pt',
    ]
    
    print(f"\nChecking S3 cache: s3://{bucket}/{prefix}/")
    print(f"Cache directory: {cache_dir}/")
    
    downloaded = 0
    skipped = 0
    
    for filename in files_to_download:
        local_path = os.path.join(cache_dir, filename)
        s3_key = f"{prefix}/{filename}"
        
        # Check if file exists locally
        if os.path.exists(local_path) and not force_download:
            # Check if S3 file is newer than local cache
            try:
                s3_response = s3.head_object(Bucket=bucket, Key=s3_key)
                s3_last_modified = s3_response['LastModified']
                
                # Get local file modification time
                local_mtime = os.path.getmtime(local_path)
                local_mtime_dt = datetime.datetime.fromtimestamp(local_mtime, tz=s3_last_modified.tzinfo)
                
                # Download if S3 file is newer
                if s3_last_modified > local_mtime_dt:
                    print(f"  ↻ S3 file newer, re-downloading: {filename}")
                    # Continue to download section below
                else:
                    skipped += 1
                    print(f"  ✓ Using cached: {filename}")
                    continue
            except s3.exceptions.ClientError:
                # File doesn't exist on S3 or error, use local cache
                print(f"  ⚠ Using local cache (S3 check failed): {filename}")
                continue
        
        # Download from S3
        try:
            print(f"  ↓ Downloading: {filename}...")
            s3.download_file(bucket, s3_key, local_path)
            downloaded += 1
            print(f"  ✓ Downloaded: {filename}")
        except s3.exceptions.ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == '404':
                print(f"  ✗ File not found on S3: {s3_key}")
                raise FileNotFoundError(f"Required file not found on S3: s3://{bucket}/{s3_key}")
            else:
                print(f"  ✗ Error downloading {filename}: {e}")
                raise
    
    if downloaded > 0:
        print(f"\nDownloaded {downloaded} files, skipped {skipped} (using cache)")
    else:
        print(f"\nAll files already cached ({skipped} files)")
    
    return cache_dir

