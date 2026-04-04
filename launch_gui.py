#!/usr/bin/env python
"""
launch_gui.py - Simple launcher for FedGuard GUI
"""
import subprocess
import sys
import os

def launch_gui():
    """Launch the Streamlit GUI"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(script_dir, 'app.py')

    print("🚀 Launching FedGuard GUI...")
    print(f"📁 App path: {app_path}")
    print("🌐 Open http://localhost:8501 in your browser")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)

    try:
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', app_path],
                      cwd=script_dir, check=True)
    except KeyboardInterrupt:
        print("\n👋 GUI stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch GUI: {e}")

if __name__ == "__main__":
    launch_gui()