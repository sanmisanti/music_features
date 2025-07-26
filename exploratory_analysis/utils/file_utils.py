"""
File Utils Module

Utilities for file operations, size formatting, and system resource management.
"""

import os
import psutil
from pathlib import Path
from typing import Union, Optional

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string (e.g., "1.2 GB", "345 MB")
    """
    if size_bytes == 0:
        return "0 B"
    
    # Define size units
    size_units = ["B", "KB", "MB", "GB", "TB", "PB"]
    
    # Calculate appropriate unit
    import math
    unit_index = min(int(math.floor(math.log(size_bytes, 1024))), len(size_units) - 1)
    
    # Calculate size in the appropriate unit
    size = size_bytes / (1024 ** unit_index)
    
    # Format with appropriate precision
    if size >= 10:
        return f"{size:.1f} {size_units[unit_index]}"
    else:
        return f"{size:.2f} {size_units[unit_index]}"

def get_memory_usage() -> float:
    """
    Get current memory usage of the process in MB.
    
    Returns:
        Memory usage in megabytes
    """
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # Convert to MB
    except Exception:
        return 0.0

def get_file_size(file_path: Union[str, Path]) -> int:
    """
    Get file size in bytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in bytes, or 0 if file doesn't exist
    """
    try:
        return Path(file_path).stat().st_size
    except (OSError, FileNotFoundError):
        return 0

def ensure_directory(directory_path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if necessary.
    
    Args:
        directory_path: Path to directory
        
    Returns:
        Path object of the directory
    """
    dir_path = Path(directory_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path

def get_available_disk_space(path: Union[str, Path] = ".") -> int:
    """
    Get available disk space in bytes.
    
    Args:
        path: Path to check (default: current directory)
        
    Returns:
        Available space in bytes
    """
    try:
        statvfs = os.statvfs(path)
        return statvfs.f_frsize * statvfs.f_bavail
    except (AttributeError, OSError):
        # Fallback for Windows
        try:
            import shutil
            return shutil.disk_usage(path).free
        except Exception:
            return 0

def is_file_readable(file_path: Union[str, Path]) -> bool:
    """
    Check if file exists and is readable.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if file is readable, False otherwise
    """
    try:
        path = Path(file_path)
        return path.exists() and path.is_file() and os.access(path, os.R_OK)
    except Exception:
        return False

def get_file_extension(file_path: Union[str, Path]) -> str:
    """
    Get file extension (without the dot).
    
    Args:
        file_path: Path to the file
        
    Returns:
        File extension string (lowercase)
    """
    return Path(file_path).suffix.lower().lstrip('.')

def safe_filename(filename: str) -> str:
    """
    Convert string to safe filename by removing/replacing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename string
    """
    import re
    # Remove invalid characters
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove multiple underscores
    safe_name = re.sub(r'_+', '_', safe_name)
    # Strip leading/trailing underscores and spaces
    return safe_name.strip('_ ')

class FileUtils:
    """Legacy class for backwards compatibility."""
    
    def __init__(self):
        pass
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        return format_file_size(size_bytes)
    
    @staticmethod
    def get_memory_usage() -> float:
        return get_memory_usage()
    
    @staticmethod
    def get_file_size(file_path: Union[str, Path]) -> int:
        return get_file_size(file_path)