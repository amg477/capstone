#!/usr/bin/env python3
"""
Run script for Influence Explorer
This calls the main app.py file
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit application."""
    print("🚀 Starting Influence Explorer...")
    print("   Running: streamlit run app.py")
    print("   Open: http://localhost:8501")
    print("-" * 40)
    
    # Set environment variables
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped")

if __name__ == "__main__":
    main()