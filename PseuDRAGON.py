"""Main entry point for PseuDRAGON."""


import os
import sys

# Add the current directory to Python path

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Use UTF-8 for Unicode output on Windows.

if sys.platform == "win32":
    try:
        # Reconfigure stdout and stderr to use UTF-8 encoding

        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception as e:
        print(f"Warning: Could not set UTF-8 encoding: {e}")


from web_interface.app import AppConfig, app

if __name__ == "__main__":
    app.run(
        debug=AppConfig.DEBUG,
        host=AppConfig.HOST,
        port=AppConfig.PORT,
        use_reloader=False,
        threaded=True,
    )
