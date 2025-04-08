#!/bin/bash

# Function to clean up test directories
cleanup_test_dirs() {
    echo "Cleaning up test directories..."
    
    # Find and remove all tmp_* directories
    find . -type d -name "tmp_*" -exec rm -rf {} + 2>/dev/null
    
    # Also clean up any .pytest_cache directories
    find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null
    
    echo "Cleanup completed at $(date)"
}

# Main loop
while true; do
    cleanup_test_dirs
    # Sleep for 10 minutes before next cleanup
    sleep 600
done 