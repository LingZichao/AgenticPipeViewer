#!/usr/bin/env python3
"""
AgentPipeViewer Web Interface Launcher

This script automatically starts a local HTTP server and opens the web interface
in the default browser.
"""

import os
import sys
import time
import socket
import subprocess
import webbrowser
from pathlib import Path

def find_available_port(start_port=8000, max_port=8999):
    """Find an available port starting from start_port"""
    for port in range(start_port, max_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(('localhost', port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No available ports found between {start_port}-{max_port}")

def main():
    print("Starting AgentPipeViewer Web Interface...")
    print("=" * 50)

    # Get the directory where this script is located
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)

    # Check if the HTML file exists
    html_file = script_dir / "pipeline_viewer.html"
    if not html_file.exists():
        print("Error: pipeline_viewer.html not found in current directory")
        print("Please run this script from the AgenticPipeViewer directory")
        sys.exit(1)

    # Find an available port
    try:
        port = find_available_port()
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    url = f"http://localhost:{port}/pipeline_viewer.html"

    print(f"Starting HTTP server on port {port}...")
    print(f"Web interface will be available at: {url}")
    print()
    print("Press Ctrl+C to stop the server")
    print()

    # Open browser after a short delay
    def open_browser():
        time.sleep(2)
        try:
            webbrowser.open(url)
        except Exception as e:
            print(f"Could not open browser automatically: {e}")
            print(f"Please open your browser and navigate to: {url}")

    # Start browser opener in background
    import threading
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()

    # Start the HTTP server
    try:
        # Use python -m http.server for better compatibility
        cmd = [sys.executable, "-m", "http.server", str(port)]
        subprocess.run(cmd, cwd=script_dir)
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()