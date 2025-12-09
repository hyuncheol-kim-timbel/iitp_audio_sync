#!/usr/bin/env python3
"""
오디오 동기화 배치 처리 스크립트 (7채널 마이크 시스템)

지정된 디렉토리를 재귀적으로 탐색하여 다음 패턴의 폴더 쌍을 찾습니다:
- 프로젝트 (7채널 .aup3 파일 폴더) + 음성_360 (비교 대상 WAV 파일 폴더)

프로젝트 폴더의 .aup3 파일을 7채널로 추출하고,
중앙 마이크 채널과 음성_360을 비교하여 모든 채널에 동일한 오프셋을 적용합니다.
"""

__version__ = "1.0.2"

import os
import subprocess
import sys
import glob
from pathlib import Path
import time
from logger_utils import ProcessLogger


def is_reference_folder(folder_path):
    """
    폴더가 기준 폴더인지 확인 (JSON 파일 포함 여부)

    Args:
        folder_path: 확인할 폴더 경로

    Returns:
        bool: JSON 파일이 있으면 True
    """
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    return len(json_files) > 0


def find_audio_folder_pairs(base_dir="."):
    """
    재귀적으로 폴더를 탐색하여 "프로젝트 + 음성_360" 패턴의 폴더 쌍을 찾습니다.

    Args:
        base_dir: 탐색 시작 디렉토리

    Returns:
        List[tuple]: [(parent_dir, project_folder, wav_folder, aup3_count, wav_count), ...]
    """
    pairs = []

    for root, dirs, files in os.walk(base_dir):
        # "프로젝트"와 "음성_360" 폴더가 둘 다 있는지 확인
        if "프로젝트" in dirs and "음성_360" in dirs:
            project_path = os.path.join(root, "프로젝트")
            wav_path = os.path.join(root, "음성_360")

            # 프로젝트 폴더에 .aup3 파일 확인
            aup3_files = glob.glob(os.path.join(project_path, "*.aup3"))

            # 음성_360 폴더에 WAV 파일 확인
            wav_files = glob.glob(os.path.join(wav_path, "*.wav"))

            if aup3_files and wav_files:
                pairs.append((root, "프로젝트", "음성_360", len(aup3_files), len(wav_files)))

    return pairs


def check_folder_structure(project_folder, wav_folder):
    """
    폴더 쌍이 올바른 구조를 가지고 있는지 확인합니다.

    Args:
        project_folder: 프로젝트 폴더 경로 (.aup3 파일)
        wav_folder: 음성_360 폴더 경로 (WAV 파일)

    Returns:
        tuple: (is_valid, message)
    """
    if not os.path.exists(project_folder):
        return False, f"프로젝트 폴더를 찾을 수 없습니다: {project_folder}"

    if not os.path.exists(wav_folder):
        return False, f"음성_360 폴더를 찾을 수 없습니다: {wav_folder}"

    # 프로젝트 폴더에 .aup3 파일 확인
    aup3_files = glob.glob(os.path.join(project_folder, "*.aup3"))

    # 음성_360 폴더에 WAV 파일 확인
    wav_files = glob.glob(os.path.join(wav_folder, "*.wav"))

    if not aup3_files:
        return False, f"프로젝트 폴더에 .aup3 파일이 없습니다: {project_folder}"

    if not wav_files:
        return False, f"음성_360 폴더에 WAV 파일이 없습니다: {wav_folder}"

    return True, f"유효함 (프로젝트: {len(aup3_files)}개 .aup3, 음성_360: {len(wav_files)}개 WAV)"


def process_folder_pair(parent_dir, project_folder_name, wav_folder_name, aup3_count, wav_count, logger=None, only_with_sync=False, failed_files_log=None):
    """
    하나의 폴더 쌍을 처리합니다.

    Args:
        parent_dir: 부모 디렉토리
        project_folder_name: 프로젝트 폴더 이름 ("프로젝트")
        wav_folder_name: WAV 폴더 이름 ("음성_360")
        aup3_count: 프로젝트 폴더의 .aup3 파일 수
        wav_count: 음성_360 폴더의 WAV 파일 수
        logger: ProcessLogger 인스턴스 (선택)
        failed_files_log: 실패 파일 로그 경로 (선택)

    Returns:
        tuple: (success, message, elapsed_time, failed_count)
    """
    # 전체 경로 구성
    project_folder = os.path.join(parent_dir, project_folder_name)
    wav_folder = os.path.join(parent_dir, wav_folder_name)

    # 폴더 구조 확인
    is_valid, msg = check_folder_structure(project_folder, wav_folder)
    if not is_valid:
        if logger:
            logger.log_folder_pair_result(False, msg)
        else:
            print(f"[건너뛰기] {msg}")
        return False, msg, 0, 0

    # audio_sync.py 실행
    try:
        start_time = time.time()

        if logger:
            logger.log_step("시작", "audio_sync.py 실행...")
        else:
            print(f"\n[시작] audio_sync.py 실행...\n")

        # .sync 파일 체크 (프로젝트 폴더에서)
        project_folder = os.path.join(parent_dir, project_folder_name)
        sync_files = glob.glob(os.path.join(project_folder, "*.sync"))
        sync_hint_path = sync_files[0] if sync_files else None

        # Python 스크립트 실행 (parent_dir 전달)
        # 로거가 있으면 로그 파일 경로도 전달
        cmd = [sys.executable, "audio_sync.py", parent_dir]
        if logger:
            cmd.extend(["--log-file", logger.log_file])
        if only_with_sync:
            cmd.append("--only-with-sync")
        if sync_hint_path:
            cmd.extend(["--sync-hint", sync_hint_path])
            if logger:
                logger.log_step("힌트", f".sync 파일 발견: {os.path.basename(sync_hint_path)}")
            else:
                print(f"[힌트] .sync 파일 발견: {os.path.basename(sync_hint_path)}")

        result = subprocess.run(
            cmd,
            encoding='utf-8',
            errors='replace'
        )

        elapsed_time = time.time() - start_time

        # 실패 파일 수집 (.failed_files.tmp 확인)
        failed_count = 0
        failed_files_tmp = os.path.join(project_folder, ".failed_files.tmp")
        if os.path.exists(failed_files_tmp):
            try:
                with open(failed_files_tmp, 'r', encoding='utf-8') as f:
                    failed_lines = f.readlines()
                    failed_count = len(failed_lines)

                    # 배치 실패 파일 로그에 추가
                    if failed_files_log:
                        with open(failed_files_log, 'a', encoding='utf-8') as log_f:
                            for line in failed_lines:
                                log_f.write(line)

                # 임시 파일 삭제
                os.remove(failed_files_tmp)
            except Exception as e:
                if logger:
                    logger.log_warning(f"실패 파일 수집 오류: {e}", indent=0)
                else:
                    print(f"[경고] 실패 파일 수집 오류: {e}")

        if result.returncode == 0:
            if logger:
                logger.log_folder_pair_result(True, "성공", elapsed_time)
            else:
                print(f"\n[완료] 처리 성공 (소요 시간: {elapsed_time:.1f}초)")
            return True, f"성공 ({elapsed_time:.1f}초)", elapsed_time, failed_count
        else:
            error_msg = f"실패 (returncode: {result.returncode})"
            if logger:
                logger.log_folder_pair_result(False, error_msg)
            else:
                print(f"\n[오류] 처리 {error_msg}")
            return False, error_msg, elapsed_time, failed_count

    except Exception as e:
        error_msg = f"예외: {str(e)}"
        if logger:
            logger.log_error("처리 중 예외 발생", e, indent=0)
            logger.log_folder_pair_result(False, error_msg)
        else:
            print(f"\n[예외] 처리 중 오류 발생: {str(e)}")
        return False, error_msg, 0, 0


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(
        description='오디오 동기화 배치 처리 프로그램 (유연한 폴더 구조)',
        epilog='예시: python batch_process.py Test  (Test 폴더 하위를 재귀 탐색)'
    )
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    parser.add_argument('base_dir', nargs='?', default='.',
                        help='탐색할 기본 디렉토리 (기본값: 현재 디렉토리)')
    parser.add_argument('--no-log', action='store_true',
                        help='로그 파일 생성 안 함 (콘솔 출력만)')
    parser.add_argument('--only-with-sync', action='store_true',
                        help='.sync 파일 쌍이 있는 항목만 처리 (재처리용)')

    args = parser.parse_args()
    base_dir = args.base_dir

    # 로거 초기화
    logger = None if args.no_log else ProcessLogger(log_dir="logs", log_prefix="batch_process")

    # 실패 파일 로그 생성
    failed_files_log = None
    if not args.no_log:
        os.makedirs("logs", exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        failed_files_log = os.path.join("logs", f"failed_files_{timestamp}.txt")
        # 헤더 작성
        with open(failed_files_log, 'w', encoding='utf-8') as f:
            f.write(f"# 실패한 파일 목록 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# 형식: 폴더\\파일명<TAB>실패 원인\n")
            f.write("#" + "=" * 68 + "\n\n")

    if logger:
        logger.log_start("오디오 동기화 배치 처리 (7채널)")
        logger.logger.info(f"탐색 디렉토리: {os.path.basename(os.path.abspath(base_dir))}")
        if failed_files_log:
            logger.logger.info(f"실패 파일 로그: {failed_files_log}")
        logger.log_separator()
    else:
        print("=" * 70)
        print("오디오 동기화 배치 처리 (7채널)")
        print("=" * 70)
        print(f"탐색: {os.path.basename(os.path.abspath(base_dir))}")
        print("=" * 70)

    # 폴더 쌍 찾기
    if logger:
        logger.log_step("검색", "폴더 쌍 검색 중...")
    else:
        print("\n[검색] 폴더 쌍 검색 중...")
    folder_pairs = find_audio_folder_pairs(base_dir)

    # --only-with-sync: .sync 쌍이 있는 폴더만 필터링
    if args.only_with_sync:
        filtered_pairs = []
        for parent, project, wav, aup3_cnt, wav_cnt in folder_pairs:
            project_path = os.path.join(parent, project)
            wav_path = os.path.join(parent, wav)
            # 프로젝트와 음성_360 모두에 .sync 파일이 있는지 확인
            project_sync = glob.glob(os.path.join(project_path, "*.sync"))
            wav_sync = glob.glob(os.path.join(wav_path, "*.sync"))
            if project_sync and wav_sync:
                filtered_pairs.append((parent, project, wav, aup3_cnt, wav_cnt))

        if logger:
            logger.log_info(f"[필터링] .sync 쌍이 있는 {len(filtered_pairs)}/{len(folder_pairs)}개 폴더만 처리", indent=0)
        else:
            print(f"[필터링] .sync 쌍이 있는 {len(filtered_pairs)}/{len(folder_pairs)}개 폴더만 처리")
        folder_pairs = filtered_pairs

    if not folder_pairs:
        error_msg = "처리할 폴더 쌍을 찾을 수 없습니다."
        if logger:
            logger.log_error(error_msg, indent=0)
            logger.log_info("다음 조건을 확인하세요:", indent=0)
            logger.log_info("  - '프로젝트' 및 '음성_360' 폴더가 같은 위치에 있는지", indent=0)
            logger.log_info("  - 프로젝트 폴더에 .aup3 파일이 있는지", indent=0)
            logger.log_info("  - 음성_360 폴더에 WAV 파일이 있는지", indent=0)
        else:
            print(f"[오류] {error_msg}")
            print("\n다음 조건을 확인하세요:")
            print("  - '프로젝트' 및 '음성_360' 폴더가 같은 위치에 있는지")
            print("  - 프로젝트 폴더에 .aup3 파일이 있는지")
            print("  - 음성_360 폴더에 WAV 파일이 있는지")
        return 1

    if logger:
        logger.log_info(f"[발견] {len(folder_pairs)}개의 폴더 쌍을 찾았습니다:", indent=0)
        for idx, (parent, project, wav, aup3_cnt, wav_cnt) in enumerate(folder_pairs, 1):
            rel_parent = os.path.relpath(parent)
            logger.log_info(f"{idx}. {rel_parent}/", indent=2)
            logger.log_info(f"   - 프로젝트: {aup3_cnt}개 .aup3 (7채널)", indent=2)
            logger.log_info(f"   - 음성_360: {wav_cnt}개 WAV", indent=2)
    else:
        print(f"[발견] {len(folder_pairs)}개의 폴더 쌍을 찾았습니다:\n")
        for idx, (parent, project, wav, aup3_cnt, wav_cnt) in enumerate(folder_pairs, 1):
            rel_parent = os.path.relpath(parent)
            print(f"  {idx}. {rel_parent}/")
            print(f"     - 프로젝트: {aup3_cnt}개 .aup3 (7채널)")
            print(f"     - 음성_360: {wav_cnt}개 WAV")

    # 사용자 확인
    if logger:
        logger.logger.info(f"\n총 {len(folder_pairs)}개의 폴더 쌍을 처리합니다.")
    else:
        print(f"\n총 {len(folder_pairs)}개의 폴더 쌍을 처리합니다.")
    response = input("계속하시겠습니까? (y/n): ").strip().lower()

    if response != 'y':
        if logger:
            logger.logger.info("작업이 취소되었습니다.")
        else:
            print("작업이 취소되었습니다.")
        return 0

    # 각 폴더 쌍 처리
    results = {}
    total_failed_files = 0  # 전체 실패 파일 수

    for idx, (parent, project, wav, aup3_cnt, wav_cnt) in enumerate(folder_pairs, 1):
        rel_parent = os.path.relpath(parent)
        pair_name = f"{rel_parent} (프로젝트+음성_360)"

        if logger:
            logger.log_folder_pair_start(idx, len(folder_pairs), parent, project, wav, aup3_cnt, wav_cnt)
        else:
            print(f"\n\n진행: [{idx}/{len(folder_pairs)}]")

        success, message, elapsed, failed_count = process_folder_pair(parent, project, wav, aup3_cnt, wav_cnt, logger, args.only_with_sync, failed_files_log)
        results[pair_name] = (success, message, elapsed, failed_count)
        total_failed_files += failed_count

    # 최종 결과 출력
    if logger:
        logger.log_summary()
        logger.logger.info("\n상세 결과:")
        for pair_name, (success, message, elapsed, failed_count) in results.items():
            status = "[OK]" if success else "[FAIL]"
            fail_info = f" (파일 실패: {failed_count}개)" if failed_count > 0 else ""
            logger.logger.info(f"{status} {pair_name}: {message}{fail_info}")

        # 실패 파일 정보
        if total_failed_files > 0:
            logger.logger.info(f"\n[중요] 총 {total_failed_files}개 파일 처리 실패")
            if failed_files_log:
                logger.logger.info(f"[중요] 실패 파일 목록: {failed_files_log}")
    else:
        stats = {"success": 0, "failed": 0, "skipped": 0}
        for success, message, elapsed, failed_count in results.values():
            if success:
                stats["success"] += 1
            elif "건너뛰기" in message or "유효하지" in message:
                stats["skipped"] += 1
            else:
                stats["failed"] += 1

        print("\n\n" + "=" * 70)
        print("배치 처리 완료")
        print("=" * 70)
        print(f"총 {len(folder_pairs)}개 폴더 쌍: 성공 {stats['success']}, 실패 {stats['failed']}, 건너뜀 {stats['skipped']}")
        if total_failed_files > 0:
            print(f"실패한 파일: {total_failed_files}개")
            if failed_files_log:
                print(f"실패 파일 목록: {failed_files_log}")

        print("\n상세:")
        for pair_name, (success, message, elapsed, failed_count) in results.items():
            status = "[OK]" if success else "[FAIL]"
            fail_info = f" (파일 실패: {failed_count}개)" if failed_count > 0 else ""
            print(f"{status} {pair_name}: {message}{fail_info}")

        print("=" * 70)

    return 0 if (logger and logger.stats['failed'] == 0 and total_failed_files == 0) or (not logger and stats['failed'] == 0 and total_failed_files == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
