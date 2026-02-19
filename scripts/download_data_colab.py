#!/usr/bin/env python3
"""
Data Download Script for Google Colab
Resumable downloads from Google Drive with progress tracking and verification.
"""

import os
import sys
import hashlib
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import requests
from tqdm import tqdm
import gdown

# Add src to path for error handling imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.dataset.error_handling import (
    DownloadError,
    get_error_handler,
    get_retry_handler,
    get_resource_monitor
)

logger = logging.getLogger(__name__)
error_handler = get_error_handler()
retry_handler = get_retry_handler()
resource_monitor = get_resource_monitor()


class DownloadConfig:
    """Configuration for data downloads."""
    
    def __init__(
        self,
        download_dir: str = "./data",
        max_retries: int = 3,
        chunk_size: int = 1024 * 1024,
        timeout: int = 30,
        verify_ssl: bool = True
    ):
        self.download_dir = Path(download_dir)
        self.max_retries = max_retries
        self.chunk_size = chunk_size
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.download_dir.mkdir(parents=True, exist_ok=True)


class DriveDownloader:
    """Google Drive downloader with resumable capability."""
    
    def __init__(self, config: DownloadConfig):
        self.config = config
        
    def download_file(
        self,
        file_id: str,
        destination: str,
        checksum: Optional[str] = None,
        description: str = "Unknown file"
    ) -> Path:
        """Download a file from Google Drive with resumable capability."""
        destination_path = self.config.download_dir / destination
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting download: {description}")
        
        # Check system resources before download
        if not resource_monitor.check_disk_space(required_gb=2.0):
            raise DownloadError(
                message="Insufficient disk space for download",
                file_id=file_id,
                destination=destination,
                cause=Exception("Low disk space")
            )
        
        # Check if file already exists and matches checksum
        if destination_path.exists():
            if checksum and self._verify_checksum(destination_path, checksum):
                logger.info(f"File already exists and checksum matches: {destination}")
                return destination_path
            logger.warning(f"File exists but checksum doesn't match or no checksum provided. Re-downloading.")
        
        # Download with retries using retry handler
        try:
            return retry_handler.execute_with_retry(
                self._download_single_file,
                file_id,
                destination_path,
                checksum,
                description
            )
        except Exception as e:
            error_handler.handle_exception(
                DownloadError(
                    message=f"Failed to download {description}",
                    file_id=file_id,
                    destination=destination,
                    cause=e
                )
            )
            raise
    
    def _download_single_file(
        self,
        file_id: str,
        destination_path: Path,
        checksum: Optional[str],
        description: str
    ) -> Path:
        """Single download attempt."""
        try:
            # Use gdown for Google Drive downloads
            url = f"https://drive.google.com/uc?id={file_id}"
            
            # Download with progress bar
            with tqdm(
                total=None,
                unit='B',
                unit_scale=True,
                desc=f"Downloading {description}",
                ncols=100
            ) as pbar:
                def progress_hook(response, total=None):
                    if total:
                        pbar.total = total
                    pbar.update(len(response.content))
                
                gdown.download(
                    url,
                    str(destination_path),
                    quiet=False,
                    proxy=None,
                    output=str(destination_path),
                    fuzzy=False,
                    cookiefile=None,
                    no_cookies=False,
                    no_check_certificate=False,
                    no_progress=False,
                    use_cookies=False,
                    timeout=self.config.timeout
                )
            
            # Verify checksum if provided
            if checksum:
                if self._verify_checksum(destination_path, checksum):
                    logger.info(f"Download successful: {description}")
                    return destination_path
                else:
                    raise DownloadError(
                        message=f"Checksum mismatch for {description}",
                        file_id=file_id,
                        destination=str(destination_path)
                    )
            
            logger.info(f"Download successful: {description}")
            return destination_path
            
        except Exception as e:
            logger.error(f"Error in download attempt: {str(e)}")
            # Clean up partial download
            if destination_path.exists():
                try:
                    destination_path.unlink()
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up partial download: {cleanup_error}")
            raise
    
    def download_multiple_files(
        self,
        file_list: List[Dict[str, str]],
        checksum_dict: Optional[Dict[str, str]] = None
    ) -> Dict[str, Path]:
        """Download multiple files from Google Drive."""
        results = {}
        
        for file_info in file_list:
            file_id = file_info["id"]
            destination = file_info["destination"]
            description = file_info.get("description", destination)
            checksum = checksum_dict.get(destination) if checksum_dict else None
            
            try:
                downloaded_path = self.download_file(
                    file_id=file_id,
                    destination=destination,
                    checksum=checksum,
                    description=description
                )
                results[destination] = downloaded_path
            except Exception as e:
                logger.error(f"Failed to download {description}: {str(e)}")
                results[destination] = None
        
        return results
    
    def _verify_checksum(self, file_path: Path, expected_checksum: str) -> bool:
        """Verify file checksum."""
        try:
            sha256 = hashlib.sha256()
            with open(file_path, 'rb') as f:
                while chunk := f.read(self.config.chunk_size):
                    sha256.update(chunk)
            
            actual_checksum = sha256.hexdigest()
            return actual_checksum == expected_checksum.lower()
        except Exception as e:
            logger.error(f"Error verifying checksum: {str(e)}")
            return False


class MultiSourceDownloader:
    """Downloader supporting multiple sources (Google Drive, S3, HTTP)."""
    
    def __init__(self, config: DownloadConfig):
        self.config = config
        self.drive_downloader = DriveDownloader(config)
    
    def download_from_source(
        self,
        source_url: str,
        destination: str,
        checksum: Optional[str] = None,
        description: str = "Unknown file"
    ) -> Path:
        """Download from various sources based on URL pattern."""
        
        if source_url.startswith("https://drive.google.com/") or "/d/" in source_url:
            # Extract file ID from Google Drive URL
            file_id = self._extract_drive_file_id(source_url)
            return self.drive_downloader.download_file(
                file_id=file_id,
                destination=destination,
                checksum=checksum,
                description=description
            )
        elif source_url.startswith("https://") or source_url.startswith("http://"):
            return self._download_http_file(source_url, destination, checksum, description)
        elif source_url.startswith("s3://"):
            return self._download_s3_file(source_url, destination, checksum, description)
        else:
            raise ValueError(f"Unsupported source URL: {source_url}")
    
    def _extract_drive_file_id(self, url: str) -> str:
        """Extract Google Drive file ID from various URL formats."""
        if "/d/" in url:
            # Format: https://drive.google.com/file/d/FILE_ID/view
            parts = url.split("/d/")
            file_id = parts[1].split("/")[0]
        elif "id=" in url:
            # Format: https://drive.google.com/uc?id=FILE_ID
            parts = url.split("id=")
            file_id = parts[1].split("&")[0]
        else:
            raise ValueError(f"Cannot extract file ID from URL: {url}")
        
        return file_id
    
    def _download_http_file(
        self,
        url: str,
        destination: str,
        checksum: Optional[str],
        description: str
    ) -> Path:
        """Download file from HTTP/HTTPS source."""
        destination_path = self.config.download_dir / destination
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading HTTP file: {description}")
        
        with requests.get(url, stream=True, timeout=self.config.timeout) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            with open(destination_path, 'wb') as f, \
                 tqdm(
                     total=total_size,
                     unit='B',
                     unit_scale=True,
                     desc=f"Downloading {description}",
                     ncols=100
                 ) as pbar:
                
                for chunk in r.iter_content(chunk_size=self.config.chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        # Verify checksum if provided
        if checksum and not self.drive_downloader._verify_checksum(destination_path, checksum):
            raise ValueError(f"Checksum mismatch for {description}")
        
        return destination_path
    
    def _download_s3_file(
        self,
        url: str,
        destination: str,
        checksum: Optional[str],
        description: str
    ) -> Path:
        """Download file from S3 source."""
        try:
            import boto3
        except ImportError:
            raise ImportError("boto3 is required for S3 downloads. Install with: pip install boto3")
        
        # Parse S3 URL: s3://bucket/key
        parts = url[5:].split("/", 1)  # Remove 's3://' and split
        if len(parts) != 2:
            raise ValueError(f"Invalid S3 URL format: {url}")
        
        bucket_name = parts[0]
        object_key = parts[1]
        
        s3 = boto3.client('s3')
        destination_path = self.config.download_dir / destination
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading S3 file: {description}")
        
        s3.download_file(bucket_name, object_key, str(destination_path))
        
        # Verify checksum if provided
        if checksum:
            from botocore.client import Config
            s3 = boto3.client('s3', config=Config(signature_version='s3v4'))
            head_object = s3.head_object(Bucket=bucket_name, Key=object_key)
            etag = head_object['ETag'].strip('"')
            if etag != checksum:
                raise ValueError(f"Checksum mismatch for {description}")
        
        return destination_path


def get_colab_downloader() -> DriveDownloader:
    """Get a DriveDownloader configured for Google Colab."""
    # Colab has limited disk space, so use a reasonable download directory
    download_dir = "./data"
    
    # Configure logging for Colab
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    config = DownloadConfig(
        download_dir=download_dir,
        max_retries=3,
        chunk_size=1024 * 1024,  # 1MB chunks
        timeout=30,
        verify_ssl=True
    )
    
    return DriveDownloader(config)


if __name__ == "__main__":
    # Example usage
    downloader = get_colab_downloader()
    
    # Example: Download a test file from Google Drive
    test_file_id = "1O2s1YvG8Z3x4C5d6E7f8G9h0I1j2K3l4M"  # Replace with actual file ID
    test_destination = "test_dataset/test_file.zip"
    
    try:
        downloaded_file = downloader.download_file(
            file_id=test_file_id,
            destination=test_destination,
            description="Test Dataset"
        )
        print(f"Successfully downloaded to: {downloaded_file}")
    except Exception as e:
        print(f"Download failed: {str(e)}")
        sys.exit(1)