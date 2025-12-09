#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.sync 파일이 있는 항목만 재처리하는 스크립트
사용법: python retry_with_sync.py <검색_루트_폴더>
"""

import os
import sys
import glob
import subprocess
import argparse
from pathlib import Path

def find_sync_files(root_dir):
    """
    루트 디렉토리 아래의 모든 프로젝트 폴더에서 .sync 파일을 찾습니다.
    음성_360 폴더에서 대응하는 .sync 파일도 함께 찾습니다.

    Returns:
        list: [(parent_dir, ref_sync_file, target_sync_file), ...] 형태의 리스트
        target_sync_file은 None일 수 있음 (대상 .sync 파일이 없는 경우)
    """
    sync_items = []

    # 프로젝트 폴더 찾기
    for root, dirs, files in os.walk(root_dir):
        # 프로젝트 폴더인지 확인
        if os.path.basename(root) == "프로젝트":
            parent_dir = os.path.dirname(root)

            # 이 폴더에서 .sync 파일 찾기
            sync_files = glob.glob(os.path.join(root, "*.sync"))

            if sync_files:
                # 각 .sync 파일에 대해 대응하는 음성_360 폴더의 .sync 파일 찾기
                voice360_dir = os.path.join(parent_dir, "음성_360")

                for sync_file in sync_files:
                    # 파일명에서 인덱스 추출 (예: VID_20250701_180651_00_001.sync -> 001)
                    sync_basename = os.path.basename(sync_file)
                    # 마지막 _XXX.sync 부분에서 XXX 추출
                    import re
                    match = re.search(r'_(\d+)\.sync$', sync_basename)
                    if not match:
                        # 인덱스를 찾을 수 없으면 파일명 그대로 매칭 시도
                        sync_items.append((parent_dir, sync_file, None))
                        continue

                    index = match.group(1)

                    # 대응하는 target .sync 파일 찾기
                    target_sync_file = None
                    if os.path.exists(voice360_dir):
                        # 음성_360 폴더에서 같은 인덱스를 가진 .sync 파일 찾기
                        target_sync_files = glob.glob(os.path.join(voice360_dir, f"*_{index}.sync"))
                        if target_sync_files:
                            target_sync_file = target_sync_files[0]

                    sync_items.append((parent_dir, sync_file, target_sync_file))

    return sync_items

def main():
    parser = argparse.ArgumentParser(description='.sync 파일이 있는 항목만 재처리')
    parser.add_argument('root_dir', help='검색할 루트 폴더 경로')
    parser.add_argument('-y', '--yes', action='store_true', help='확인 없이 자동으로 진행')

    args = parser.parse_args()
    root_dir = args.root_dir

    if not os.path.exists(root_dir):
        print(f"오류: 경로를 찾을 수 없습니다: {root_dir}")
        sys.exit(1)

    print("=" * 70)
    print("동기화 재처리 (.sync 파일이 있는 항목만)")
    print("=" * 70)
    print(f"검색 경로: {root_dir}")
    print()

    # .sync 파일 찾기
    sync_items = find_sync_files(root_dir)

    if not sync_items:
        print("재처리할 항목이 없습니다. (.sync 파일을 찾을 수 없음)")
        sys.exit(0)

    print(f"총 {len(sync_items)}개의 .sync 파일을 찾았습니다:")
    for i, (parent_dir, ref_sync_file, target_sync_file) in enumerate(sync_items, 1):
        folder_name = os.path.basename(parent_dir)
        ref_sync_name = os.path.basename(ref_sync_file)
        if target_sync_file:
            target_sync_name = os.path.basename(target_sync_file)
            print(f"  {i}. {folder_name}/{ref_sync_name} + 음성_360/{target_sync_name}")
        else:
            print(f"  {i}. {folder_name}/{ref_sync_name} (대상 .sync 없음)")
    print()

    # 확인
    if not args.yes:
        response = input(f"{len(sync_items)}개 항목을 재처리하시겠습니까? (y/n): ").strip().lower()
        if response not in ['y', 'yes']:
            print("취소되었습니다.")
            sys.exit(0)

    print()
    print("=" * 70)
    print("재처리 시작")
    print("=" * 70)

    # 각 항목 처리
    success_count = 0
    fail_count = 0

    for i, (parent_dir, ref_sync_file, target_sync_file) in enumerate(sync_items, 1):
        folder_name = os.path.basename(parent_dir)
        ref_sync_name = os.path.basename(ref_sync_file)

        print()
        if target_sync_file:
            target_sync_name = os.path.basename(target_sync_file)
            print(f"[{i}/{len(sync_items)}] {folder_name} - {ref_sync_name} + {target_sync_name}")
        else:
            print(f"[{i}/{len(sync_items)}] {folder_name} - {ref_sync_name} (단일 힌트)")
        print("-" * 70)

        # audio_sync.py 실행
        cmd = [
            sys.executable,
            "audio_sync.py",
            parent_dir,
            "--sync-hint",
            ref_sync_file
        ]

        # 대상 .sync 파일이 있으면 추가
        if target_sync_file:
            cmd.extend(["--target-sync-hint", target_sync_file])

        try:
            result = subprocess.run(
                cmd,
                cwd=os.path.dirname(__file__),
                capture_output=False,
                text=True
            )

            if result.returncode == 0:
                print(f"[완료] {folder_name} 처리 성공")
                success_count += 1
            else:
                print(f"[오류] {folder_name} 처리 실패 (exit code: {result.returncode})")
                fail_count += 1

        except Exception as e:
            print(f"[오류] {folder_name} 처리 중 예외 발생: {e}")
            fail_count += 1

    print()
    print("=" * 70)
    print("재처리 완료")
    print("=" * 70)
    print(f"성공: {success_count}개")
    print(f"실패: {fail_count}개")
    print(f"전체: {len(sync_items)}개")

    if fail_count > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
