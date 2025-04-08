#!/bin/bash

# Function to clean up test directories
cleanup_test_dirs() {
    echo "Cleaning up test directories..."
    
    # Find all tmp_* directories and sort by modification time (newest first)
    # Keep the 16 most recent ones and remove all others
    find . -type d -name "tmp_*" -printf "%T@ %p\n" | sort -nr | \
    awk 'NR>16 {print $2}' | xargs -r rm -rf 2>/dev/null
    
    # Also clean up any .pytest_cache directories
    find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null
    
    echo "Cleanup completed at $(date)"
}

# Main loop
while true; do
    cleanup_test_dirs
    # Sleep for 10 minutes before next cleanup
    sleep 300
done 