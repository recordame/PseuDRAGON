"""
Main entry point for PseuDRAGON
PseuDRAGON 메인 진입점
"""

# Standard library imports
# 표준 라이브러리 import
import os
import sys

# Add the current directory to Python path
# 현재 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Fix Windows console encoding for emoji support
# Windows 콘솔 인코딩 수정 (이모지 지원)
if sys.platform == "win32":
    try:
        # Reconfigure stdout and stderr to use UTF-8 encoding
        # stdout와 stderr를 UTF-8 인코딩으로 재설정
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception as e:
        print(f"Warning: Could not set UTF-8 encoding: {e}")

# Project-specific imports
# 프로젝트 관련 import
from web_interface.app import app

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
