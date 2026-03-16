#!/bin/bash

# AgentPipeViewer Web Interface Launcher
# This script starts a local HTTP server and opens the web interface in the default browser

echo "Starting AgentPipeViewer Web Interface..."
echo "=========================================="

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed or not in PATH"
    exit 1
fi

# Check if the HTML file exists
if [ ! -f "pipeline_viewer.html" ]; then
    echo "Error: pipeline_viewer.html not found in current directory"
    echo "Please run this script from the AgenticPipeViewer directory"
    exit 1
fi

# Find an available port (starting from 8000)
PORT=8000
while lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; do
    PORT=$((PORT + 1))
    if [ $PORT -gt 8999 ]; then
        echo "Error: Could not find an available port between 8000-8999"
        exit 1
    fi
done

echo "Starting HTTP server on port $PORT..."
echo "Web interface will be available at: http://localhost:$PORT/pipeline_viewer.html"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Function to open browser (cross-platform)
open_browser() {
    local url=$1
    if command -v xdg-open &> /dev/null; then
        xdg-open "$url" 2>/dev/null &
    elif command -v open &> /dev/null; then
        open "$url" 2>/dev/null &
    elif command -v start &> /dev/null; then
        start "$url" 2>/dev/null &
    else
        echo "Please open your browser and navigate to: $url"
    fi
}

# Open browser after a short delay
(sleep 2 && open_browser "http://localhost:$PORT/pipeline_viewer.html") &

# Start the HTTP server
python3 -m http.server $PORT

echo ""
echo "Server stopped."