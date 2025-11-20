#!/usr/bin/env python3
"""
로깅 유틸리티 모듈
오디오 동기화 처리 과정의 상세 로그를 파일로 기록합니다.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path


class ProcessLogger:
    """처리 과정을 상세하게 로그로 기록하는 클래스"""

    def __init__(self, log_dir="logs", log_prefix="audio_sync"):
        """
        로거 초기화

        Args:
            log_dir: 로그 파일을 저장할 디렉토리
            log_prefix: 로그 파일명 접두사
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # 타임스탬프를 포함한 로그 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"{log_prefix}_{timestamp}.log")

        # 로거 설정
        self.logger = logging.getLogger(f"{log_prefix}_{timestamp}")
        self.logger.setLevel(logging.DEBUG)

        # 파일 핸들러 (상세 로그)
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)

        # 콘솔 핸들러 (기본 정보만)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # 처리 통계
        self.stats = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "start_time": datetime.now()
        }

    def log_separator(self, char="=", length=70):
        """구분선 출력"""
        separator = char * length
        self.logger.info(separator)

    def log_header(self, title):
        """헤더 출력"""
        self.log_separator()
        self.logger.info(title)
        self.log_separator()

    def log_start(self, title):
        """처리 시작 로그"""
        self.log_header(title)
        self.logger.info(f"시작 시간: {self.stats['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"로그 파일: {self.log_file}")
        self.log_separator()

    def log_folder_pair_start(self, pair_index, total_pairs, parent_dir, project_folder, wav_folder, aup3_count, wav_count):
        """폴더 쌍 처리 시작"""
        self.log_separator()
        self.logger.info(f"[{pair_index}/{total_pairs}] 폴더 쌍 처리 시작")
        self.log_separator()
        self.logger.info(f"작업 폴더: {parent_dir}")
        self.logger.info(f"  - 프로젝트: {project_folder} ({aup3_count}개 .aup3)")
        self.logger.info(f"  - 음성_360: {wav_folder} ({wav_count}개 WAV)")
        self.stats["total"] += 1

    def log_file_processing_start(self, file_index, total_files, aup3_file, wav_file):
        """파일 처리 시작"""
        self.logger.info("")
        self.logger.info(f"[{file_index}/{total_files}] 파일 처리 시작")
        self.logger.info(f"  작업 파일:")
        self.logger.info(f"    - .aup3: {aup3_file}")
        self.logger.info(f"    - WAV: {wav_file}")

    def log_step(self, step_name, message=""):
        """처리 단계 로그"""
        if message:
            self.logger.info(f"  [{step_name}] {message}")
        else:
            self.logger.info(f"  [{step_name}]")

    def log_info(self, message, indent=2):
        """정보 로그"""
        prefix = " " * indent
        self.logger.info(f"{prefix}→ {message}")

    def log_warning(self, message, indent=2):
        """경고 로그"""
        prefix = " " * indent
        self.logger.warning(f"{prefix}[경고] {message}")

    def log_error(self, message, error=None, indent=2):
        """오류 로그"""
        prefix = " " * indent
        self.logger.error(f"{prefix}[오류] {message}")
        if error:
            self.logger.error(f"{prefix}  상세: {str(error)}")
            import traceback
            self.logger.debug(traceback.format_exc())

    def log_success(self, message, elapsed_time=None, indent=2):
        """성공 로그"""
        prefix = " " * indent
        if elapsed_time:
            self.logger.info(f"{prefix}[완료] {message} (소요 시간: {elapsed_time:.1f}초)")
        else:
            self.logger.info(f"{prefix}[완료] {message}")

    def log_offset(self, offset_seconds, offset_samples):
        """오프셋 정보 로그"""
        self.log_info(f"계산된 오프셋: {offset_seconds:.3f}초 ({offset_samples} 샘플)", indent=4)

    def log_trim_info(self, trim_left, trim_right, sample_rate):
        """Trim 정보 로그"""
        self.log_info(f"Trim 범위 감지:", indent=4)
        self.log_info(f"  trimLeft: {trim_left} 샘플 ({trim_left/sample_rate:.2f}초)", indent=6)
        self.log_info(f"  trimRight: {trim_right} 샘플 ({trim_right/sample_rate:.2f}초)", indent=6)

    def log_channel_sync(self, channel_idx, total_channels, channel_name):
        """채널 동기화 로그"""
        self.log_info(f"[{channel_idx}/{total_channels}] {channel_name} 동기화 중...", indent=4)

    def log_json_processing(self, json_file, offset_seconds):
        """JSON 파일 처리 로그"""
        self.log_info(f"JSON 파일 발견: {json_file}", indent=4)
        self.log_info(f"타임스탬프에 오프셋 {offset_seconds:.3f}초 적용 중...", indent=4)

    def log_file_copy(self, source_name, dest_dir):
        """파일 복사 로그"""
        self.log_info(f"파일 복사: {source_name} → {dest_dir}", indent=4)

    def log_folder_pair_result(self, success, message, elapsed_time=None):
        """폴더 쌍 처리 결과"""
        if success:
            self.stats["success"] += 1
            if elapsed_time:
                self.log_success(f"폴더 쌍 처리 성공 ({elapsed_time:.1f}초)", indent=0)
            else:
                self.log_success("폴더 쌍 처리 성공", indent=0)
        else:
            if "건너뛰기" in message or "유효하지" in message:
                self.stats["skipped"] += 1
                self.logger.info(f"[건너뛰기] {message}")
            else:
                self.stats["failed"] += 1
                self.log_error(f"폴더 쌍 처리 실패: {message}", indent=0)

    def log_file_result(self, success, message, elapsed_time=None):
        """파일 처리 결과"""
        if success:
            if elapsed_time:
                self.log_success(f"파일 처리 성공 ({elapsed_time:.1f}초)", indent=2)
            else:
                self.log_success("파일 처리 성공", indent=2)
        else:
            self.log_error(f"파일 처리 실패: {message}", indent=2)

    def log_summary(self):
        """최종 요약"""
        end_time = datetime.now()
        elapsed_total = (end_time - self.stats["start_time"]).total_seconds()

        self.logger.info("")
        self.log_separator()
        self.logger.info("처리 완료 요약")
        self.log_separator()
        self.logger.info(f"총 처리: {self.stats['total']}개 폴더 쌍")
        self.logger.info(f"  - 성공: {self.stats['success']}개")
        self.logger.info(f"  - 실패: {self.stats['failed']}개")
        self.logger.info(f"  - 건너뜀: {self.stats['skipped']}개")
        self.logger.info(f"전체 소요 시간: {elapsed_total:.1f}초")
        self.logger.info(f"종료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_separator()
        self.logger.info(f"상세 로그: {self.log_file}")
        self.log_separator()

    def get_stats(self):
        """통계 반환"""
        return self.stats.copy()
