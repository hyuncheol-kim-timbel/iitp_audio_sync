"""nul 파일 삭제 스크립트"""
import os
from pathlib import Path

# 현재 디렉토리의 nul 파일 삭제
current_dir = Path(__file__).parent
nul_file = current_dir / "nul"

if nul_file.exists():
    try:
        # Windows 예약어 파일 삭제를 위한 UNC 경로 사용
        unc_path = f"\\\\?\\{nul_file.resolve()}"
        os.remove(unc_path)
        print(f"[OK] nul file deleted: {nul_file}")
    except Exception as e:
        print(f"[ERROR] Failed to delete nul file: {e}")
        print(f"\nManual deletion:")
        print(f'  CMD: del "\\\\?\\{nul_file.resolve()}"')
        print(f'  PowerShell: Remove-Item "\\\\?\\{nul_file.resolve()}"')
else:
    print("nul file not found.")
