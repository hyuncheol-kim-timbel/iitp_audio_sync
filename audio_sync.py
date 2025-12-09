#!/usr/bin/env python3
"""
오디오 동기화 메인 프로그램

두 개의 마이크로 녹음된 오디오 파일들을 동기화합니다.
JSON 파일을 포함한 폴더를 기준으로 다른 폴더의 오디오를 동기화합니다.
"""

__version__ = "1.0.2"

import os
import sys
import argparse
from pathlib import Path
import config
from sync_logic import (
    load_audio,
    calculate_time_offset,
    sync_audio_pair,
    find_matching_files,
    find_aup3_and_wav_pairs,
    adjust_json_timestamps,
    find_json_file,
    parse_sync_file
)
from aup3_converter import extract_all_tracks_from_aup3
import glob
import tempfile
import shutil
import time
import logging


def setup_logger(log_file=None):
    """
    로거 설정

    Args:
        log_file: 로그 파일 경로 (None이면 콘솔만 출력)

    Returns:
        logging.Logger: 설정된 로거
    """
    logger = logging.getLogger('audio_sync')
    logger.setLevel(logging.DEBUG)

    # 기존 핸들러 제거
    logger.handlers = []

    # 포맷 설정
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 파일 핸들러 (로그 파일이 지정된 경우)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


def main():
    parser = argparse.ArgumentParser(
        description='오디오 동기화 프로그램',
        epilog='예시: python audio_sync.py Test  (Test 폴더 안의 음성, 음성_360 처리)'
    )
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    parser.add_argument('parent_dir', nargs='?', default=None,
                        help='부모 폴더 경로 (생략 시 현재 디렉토리)')
    parser.add_argument('--reference', default=None,
                        help='기준 폴더 (JSON 파일 포함) - 직접 지정')
    parser.add_argument('--target', default=None,
                        help='동기화할 대상 폴더 - 직접 지정')
    parser.add_argument('--sync-duration', type=int, default=config.SYNC_DURATION,
                        help='동기화에 사용할 초기 오디오 길이 (초)')
    parser.add_argument('--sample-rate', type=int, default=config.SAMPLE_RATE,
                        help='동기화 분석용 샘플링 레이트 (Hz, 출력 파일은 원본 유지)')
    parser.add_argument('--log-file', default=None,
                        help='로그 파일 경로 (배치 처리용)')
    parser.add_argument('--sync-hint', default=None,
                        help='.sync 파일 경로 (수동 박수 위치 힌트 - 기준 오디오)')
    parser.add_argument('--target-sync-hint', default=None,
                        help='.sync 파일 경로 (수동 박수 위치 힌트 - 대상 오디오)')
    parser.add_argument('--only-with-sync', action='store_true',
                        help='.sync 파일 쌍이 있는 항목만 처리 (재처리용)')

    args = parser.parse_args()

    # 로거 설정
    logger = setup_logger(args.log_file)

    # 폴더 경로 결정
    if args.reference and args.target:
        # --reference, --target이 명시된 경우 직접 사용
        reference_dir = args.reference
        target_dir = args.target
    elif args.parent_dir:
        # parent_dir이 주어진 경우 그 안에서 찾기
        reference_dir = os.path.join(args.parent_dir, config.AUDIO_FOLDERS['reference'])
        target_dir = os.path.join(args.parent_dir, config.AUDIO_FOLDERS['target'])
    else:
        # 둘 다 없으면 현재 디렉토리에서 기본 폴더 사용
        reference_dir = config.AUDIO_FOLDERS['reference']
        target_dir = config.AUDIO_FOLDERS['target']

    sync_duration = args.sync_duration
    sr = args.sample_rate

    # .sync 힌트 파일 파싱 (기준 오디오)
    ref_hint_time = None
    if args.sync_hint and os.path.exists(args.sync_hint):
        ref_hint_time = parse_sync_file(args.sync_hint)
        if ref_hint_time is not None:
            logger.info(f"[힌트] 기준 .sync 파일에서 박수 위치 로드: {ref_hint_time:.3f}초")

    # .sync 힌트 파일 파싱 (대상 오디오)
    target_hint_time = None
    if args.target_sync_hint and os.path.exists(args.target_sync_hint):
        target_hint_time = parse_sync_file(args.target_sync_hint)
        if target_hint_time is not None:
            logger.info(f"[힌트] 대상 .sync 파일에서 박수 위치 로드: {target_hint_time:.3f}초")

    # 디렉토리 존재 확인
    if not os.path.exists(reference_dir):
        logger.error(f"[오류] 기준 폴더를 찾을 수 없습니다: {reference_dir}")
        return

    if not os.path.exists(target_dir):
        logger.error(f"[오류] 대상 폴더를 찾을 수 없습니다: {target_dir}")
        return

    logger.info("\n" + "=" * 70)
    logger.info("오디오 동기화 프로그램 (7채널 마이크)")
    logger.info("=" * 70)
    logger.info(f"프로젝트: {os.path.basename(reference_dir)}")
    logger.info(f"음성_360: {os.path.basename(target_dir)}")
    logger.info(f"분석: {sync_duration}초, {sr}Hz, 중앙ch{config.CENTER_MIC_CHANNEL}")
    logger.info("=" * 70)

    # 1. 프로젝트 폴더에서 .aup3 파일 찾기
    logger.info("\n[단계 1] .aup3 파일 및 WAV 파일 검색 중...")
    aup3_files = glob.glob(os.path.join(reference_dir, "*.aup3"))

    if not aup3_files:
        logger.error(f"[오류] 프로젝트 폴더에 .aup3 파일이 없습니다: {reference_dir}")
        return

    # 2. 음성_360과 매칭 확인
    matches = find_aup3_and_wav_pairs(reference_dir, target_dir)

    if not matches:
        logger.error("[오류] 매칭되는 .aup3 ↔ WAV 파일 쌍을 찾을 수 없습니다.")
        return

    # --only-with-sync 옵션: .sync 쌍이 있는 항목만 필터링
    if args.only_with_sync:
        filtered_matches = []
        for aup3_file, wav_file, file_index in matches:
            # 프로젝트 폴더에서 .sync 파일 찾기
            ref_sync_pattern = os.path.join(reference_dir, f"*_{file_index}.sync")
            ref_sync_files = glob.glob(ref_sync_pattern)

            # 음성_360 폴더에서 .sync 파일 찾기
            target_sync_pattern = os.path.join(target_dir, f"*_{file_index}.sync")
            target_sync_files = glob.glob(target_sync_pattern)

            # 둘 다 있는 경우만 포함
            if ref_sync_files and target_sync_files:
                filtered_matches.append((aup3_file, wav_file, file_index))

        if not filtered_matches:
            logger.warning("[경고] .sync 파일 쌍이 있는 항목이 없습니다.")
            logger.info("전체 실행: python audio_sync.py <폴더>")
            logger.info("재처리: python audio_sync.py <폴더> --only-with-sync")
            return

        logger.info(f"[필터링] .sync 쌍이 있는 {len(filtered_matches)}개 항목만 처리합니다.")
        matches = filtered_matches

    logger.info(f"[완료] {len(matches)}개의 파일 쌍을 찾았습니다.\n")

    # 3. 임시 디렉토리 생성 (7채널 추출용)
    temp_base = tempfile.mkdtemp(prefix="aup3_7ch_sync_")
    temp_ch_dir = os.path.join(temp_base, "channels")
    os.makedirs(temp_ch_dir, exist_ok=True)

    # 4. 출력 디렉토리 생성
    output_dir = reference_dir + config.OUTPUT_SUFFIX
    os.makedirs(output_dir, exist_ok=True)
    parent_folder = os.path.dirname(reference_dir)
    voice_output_dir = os.path.join(parent_folder, "음성_output")
    os.makedirs(voice_output_dir, exist_ok=True)
    logger.info(f"\n출력폴더: {os.path.basename(output_dir)}, {os.path.basename(voice_output_dir)}\n")
    logger.debug(f"출력 디렉토리 생성: {output_dir}")
    logger.debug(f"음성 출력 디렉토리 생성: {voice_output_dir}")

    # 실패 파일 목록 추적
    failed_files = []  # [(파일명, 실패 원인), ...]
    success_count = 0

    # 5. 각 .aup3 파일 처리
    for idx, (aup3_file, wav_file, file_index) in enumerate(matches, 1):
        file_start_time = time.time()

        logger.info(f"\n[{idx}/{len(matches)}] {file_index}")
        logger.info(f"  .aup3: {aup3_file}")
        logger.info(f"  WAV: {wav_file}")
        logger.debug(f"처리 시작: {aup3_file} ↔ {wav_file}")

        aup3_path = os.path.join(reference_dir, aup3_file)
        wav_path = os.path.join(target_dir, wav_file)

        try:
            # 5-1. .aup3 파일에서 모든 채널 추출
            logger.info(f"\n  [단계 1] 전체 채널 추출 중...")
            channel_files = extract_all_tracks_from_aup3(aup3_path, temp_ch_dir)

            # 음성 폴더 경로 찾기 (fallback용)
            parent_folder = os.path.dirname(reference_dir)
            voice_dir = os.path.join(parent_folder, "음성")
            base_name = Path(aup3_file).stem

            # fallback 모드 플래그
            fallback_mode = False

            if not channel_files:
                # .aup3 파싱 실패 → 음성 폴더의 WAV로 fallback
                logger.warning(f"  [경고] .aup3 채널 추출 실패 (파일 손상 가능)")
                logger.info(f"  [Fallback] 음성 폴더의 WAV 파일로 대체 처리 시도...")

                voice_wav_path = os.path.join(voice_dir, base_name + ".wav")
                if os.path.exists(voice_wav_path):
                    logger.info(f"  [Fallback] 음성 폴더 WAV 발견: {base_name}.wav")
                    fallback_mode = True
                    # aup3 손상은 반드시 실패로 기록 (Fallback으로 처리되더라도)
                    failed_files.append((aup3_file, "aup3 손상 - Fallback 처리됨"))
                    # channel_files는 사용하지 않음 (빈 상태 유지)
                else:
                    error_msg = "채널 추출 실패 + 음성 폴더 WAV 없음"
                    failed_files.append((aup3_file, error_msg))
                    logger.error(f"  [오류] {error_msg}")
                    logger.debug(f"파일 처리 실패: {aup3_file} - {error_msg}")
                    continue
            else:
                total_extracted = len(channel_files)
                logger.info(f"  [정보] {total_extracted}개 채널 추출됨")
                logger.debug(f"추출된 채널 수: {total_extracted}")

                # 사용할 채널만 선택 (1 ~ NUM_MIC_CHANNELS)
                if total_extracted < config.NUM_MIC_CHANNELS:
                    error_msg = f"필요한 {config.NUM_MIC_CHANNELS}개 채널보다 적게 추출되었습니다. (추출: {total_extracted}개)"
                    logger.error(f"  [오류] {error_msg}")
                    logger.debug(f"파일 처리 실패: {aup3_file} - {error_msg}")
                    continue

                # NUM_MIC_CHANNELS개만 사용 (나머지는 무시)
                channel_files = channel_files[:config.NUM_MIC_CHANNELS]
                logger.info(f"  [정보] {config.NUM_MIC_CHANNELS}개 채널 사용 (ch1~ch{config.NUM_MIC_CHANNELS})")
                logger.debug(f"사용할 채널: {len(channel_files)}개")

            # 5-1-1. Trim 감지 및 적용 (음성 폴더의 WAV를 기준으로) - fallback 모드에서는 건너뜀
            if not fallback_mode:
                logger.info(f"\n  [단계 1-1] Trim 범위 감지 중...")

            trim_applied = False
            trim_left = 0
            sr_trim = 0
            reference_wav_name = base_name + ".wav"
            reference_wav_path = os.path.join(voice_dir, reference_wav_name)

            # fallback 모드가 아닐 때만 Trim 처리
            if not fallback_mode:
                if os.path.exists(voice_dir):
                    if os.path.exists(reference_wav_path):
                        logger.info(f"    → 기준 WAV 파일 발견: {reference_wav_name}")
                        logger.debug(f"기준 WAV 경로: {reference_wav_path}")

                        # 중앙 마이크 채널을 사용하여 trim 범위 감지
                        # (각 채널이 Audacity에서 다른 offset을 가질 수 있으므로, 동기화에 사용되는 중앙 채널 기준)
                        from sync_logic import find_trim_range_by_reference, apply_trim_to_files

                        center_ch_idx = config.CENTER_MIC_CHANNEL - 1  # 1-based → 0-based
                        logger.info(f"    → 중앙 채널 ch{config.CENTER_MIC_CHANNEL}로 Trim 감지 중...")

                        trim_left, trim_right, sr_trim = find_trim_range_by_reference(
                            channel_files[center_ch_idx],
                            reference_wav_path,
                            downsample_factor=100
                        )

                        if trim_left is not None and trim_right is not None:
                            logger.info(f"    → 감지된 Trim 범위:")
                            logger.info(f"       trimLeft: {trim_left} 샘플 ({trim_left/sr_trim:.2f}초)")
                            logger.info(f"       trimRight: {trim_right} 샘플 ({trim_right/sr_trim:.2f}초)")
                            logger.debug(f"Trim 정보: left={trim_left}, right={trim_right}, sr={sr_trim}")

                            # 모든 채널에 동일한 trim 적용
                            logger.info(f"    → {config.NUM_MIC_CHANNELS}개 채널에 trim 적용 중...")
                            trim_success_count = apply_trim_to_files(channel_files, trim_left, trim_right)

                            if trim_success_count == config.NUM_MIC_CHANNELS:
                                logger.info(f"    → [완료] {trim_success_count}개 채널 trim 완료")
                                logger.debug(f"Trim 적용 성공: {trim_success_count}개 채널")
                                trim_applied = True
                            else:
                                logger.warning(f"    → [경고] {trim_success_count}/{config.NUM_MIC_CHANNELS}개 채널만 trim됨")
                                logger.debug(f"Trim 부분 성공: {trim_success_count}/{config.NUM_MIC_CHANNELS}")
                        else:
                            logger.warning(f"    → [경고] Trim 범위를 감지하지 못했습니다. Trim 건너뜀")
                            logger.debug("Trim 범위 감지 실패")
                    else:
                        logger.warning(f"    → [경고] 음성 폴더에서 기준 WAV를 찾을 수 없습니다: {reference_wav_name}")
                        logger.info(f"    → Trim 건너뜀 (추출된 파일 그대로 사용)")
                        logger.debug(f"기준 WAV 파일 없음: {reference_wav_path}")
                else:
                    logger.warning(f"    → [경고] 음성 폴더를 찾을 수 없습니다: {voice_dir}")
                    logger.info(f"    → Trim 건너뜀 (추출된 파일 그대로 사용)")
                    logger.debug(f"음성 폴더 없음: {voice_dir}")

            # 5-2. 중앙 마이크 채널 선택 (또는 fallback 모드에서 음성 폴더 WAV 사용)
            if fallback_mode:
                center_ch_file = reference_wav_path  # 음성 폴더의 WAV 사용
                logger.info(f"\n  [단계 2] [Fallback] 음성 폴더 WAV 사용: {reference_wav_name}")
            else:
                center_ch_file = channel_files[config.CENTER_MIC_CHANNEL - 1]  # 1-based → 0-based
                logger.info(f"\n  [단계 2] 중앙 마이크 채널 {config.CENTER_MIC_CHANNEL} 선택: {os.path.basename(center_ch_file)}")
            logger.debug(f"중앙 채널 파일: {center_ch_file}")

            # 5-3. 중앙 채널과 음성_360 비교하여 오프셋 계산
            logger.info(f"  [단계 3] 오프셋 계산 (중앙ch{config.CENTER_MIC_CHANNEL} ↔ 음성_360, {sync_duration}초)...")

            # 현재 파일에 대한 .sync 힌트 파일 자동 검색
            file_ref_hint = ref_hint_time  # 기본값은 전역 힌트
            file_target_hint = target_hint_time  # 기본값은 전역 힌트

            # 프로젝트 폴더에서 이 파일의 .sync 파일 찾기
            ref_sync_pattern = os.path.join(reference_dir, f"*_{file_index}.sync")
            ref_sync_files = glob.glob(ref_sync_pattern)
            if ref_sync_files:
                file_ref_hint = parse_sync_file(ref_sync_files[0])
                if file_ref_hint is not None:
                    logger.info(f"    → [파일별 힌트] 기준 오디오: {file_ref_hint:.3f}초 ({os.path.basename(ref_sync_files[0])})")

                    # Trim이 적용된 경우, .sync 위치를 조정해야 함
                    # .sync 파일의 위치는 원본 .aup3 기준이므로, trim_left만큼 빼야 함
                    if trim_applied and trim_left > 0 and sr_trim > 0:
                        trim_seconds = trim_left / sr_trim
                        original_hint = file_ref_hint
                        file_ref_hint = file_ref_hint - trim_seconds

                        if file_ref_hint < 0:
                            logger.warning(f"    → [경고] Trim 후 박수 위치가 음수 ({file_ref_hint:.3f}초). 자동 감지로 전환.")
                            file_ref_hint = None
                        else:
                            logger.info(f"    → [Trim 보정] 기준 힌트: {original_hint:.3f}초 → {file_ref_hint:.3f}초 (trim: {trim_seconds:.3f}초)")

            # 음성_360 폴더에서 이 파일의 .sync 파일 찾기
            target_sync_pattern = os.path.join(target_dir, f"*_{file_index}.sync")
            target_sync_files = glob.glob(target_sync_pattern)
            if target_sync_files:
                file_target_hint = parse_sync_file(target_sync_files[0])
                if file_target_hint is not None:
                    logger.info(f"    → [파일별 힌트] 대상 오디오: {file_target_hint:.3f}초 ({os.path.basename(target_sync_files[0])})")

            center_audio, sr = load_audio(center_ch_file, duration=sync_duration, sr=sr)
            target_audio, sr = load_audio(wav_path, duration=sync_duration, sr=sr)

            offset_seconds, offset_samples, sync_warning = calculate_time_offset(center_audio, target_audio, sr, ref_hint_time=file_ref_hint, target_hint_time=file_target_hint)
            logger.info(f"    → 오프셋: {offset_seconds:.3f}초 ({offset_samples} 샘플)")
            logger.debug(f"오프셋 계산 완료: offset_seconds={offset_seconds}, offset_samples={offset_samples}, sr={sr}")

            # 동기화 경고가 있으면 기록
            if sync_warning:
                logger.warning(f"  [경고] ★★★ 동기화 실패 가능: {sync_warning} ★★★")
                logger.warning(f"  [경고] ★★★ 수동 확인 필수: 박수 소리 위치가 잘못 감지되었을 수 있습니다 ★★★")
                failed_files.append((aup3_file, sync_warning))

            # 5-4. 모든 7채널에 동일한 오프셋 적용 (fallback 모드에서는 건너뜀)
            if not fallback_mode:
                logger.info(f"\n  [단계 4] {config.NUM_MIC_CHANNELS}개 채널 모두에 오프셋 적용 중...")

                for ch_idx, ch_file in enumerate(channel_files, 1):
                    ch_basename = Path(ch_file).stem  # 확장자 제거
                    output_filename = f"{ch_basename}.wav"
                    output_path = os.path.join(output_dir, output_filename)

                    logger.info(f"    [{ch_idx}/{config.NUM_MIC_CHANNELS}] ch{ch_idx} 동기화 중...")
                    logger.debug(f"채널 {ch_idx} 동기화: {ch_file} → {output_path}")

                    # 동기화 적용: 음성_360을 기준으로 7채널 조절
                    # offset이 양수면 target(음성_360)이 늦게 시작 → 7채널이 일찍 시작 → 7채널 앞부분 제거
                    # offset이 음수면 target이 일찍 시작 → 7채널이 늦게 시작 → 7채널 앞에 무음 추가
                    sync_audio_pair(wav_path, ch_file, output_path, -offset_samples, sr)
                    logger.debug(f"채널 {ch_idx} 동기화 완료: {output_filename}")

                logger.info(f"  [완료] {config.NUM_MIC_CHANNELS}개 채널 동기화 완료!\n")
            else:
                logger.info(f"\n  [단계 4] [Fallback] 7채널 동기화 건너뜀 (aup3 파싱 실패)")

            # 5-5. JSON 파일 처리 (음성 폴더에서 찾아서 프로젝트_output에 복사 및 타임스탬프 조정)
            logger.info(f"  [단계 5] JSON 파일 처리 중...")

            # voice_dir, base_name은 이미 위에서 정의됨
            json_path = None
            output_json_path = None
            json_filename = None
            audio_file_for_json = base_name + ".wav"

            if os.path.exists(voice_dir):
                # JSON 파일 찾기
                json_path = find_json_file(audio_file_for_json, voice_dir)

                if json_path:
                    json_filename = os.path.basename(json_path)
                    output_json_path = os.path.join(output_dir, json_filename)

                    logger.info(f"    → JSON 파일 발견: {json_filename}")
                    logger.info(f"    → 타임스탬프에 오프셋 {-offset_seconds:.3f}초 적용 중...")
                    logger.debug(f"JSON 파일 경로: {json_path}")
                    logger.debug(f"출력 JSON 경로: {output_json_path}")

                    # JSON 파일 복사 및 타임스탬프 조정
                    # WAV 파일과 동일하게 -offset_seconds 적용
                    # (양수면 앞부분 제거 → 타임스탬프 감소, 음수면 무음 추가 → 타임스탬프 증가)
                    adjust_json_timestamps(json_path, output_json_path, -offset_seconds)

                    logger.info(f"    → JSON 파일 저장 완료: {json_filename}")
                    logger.debug(f"JSON 타임스탬프 조정 완료: offset={-offset_seconds}초")
                else:
                    logger.warning(f"    [경고] JSON 파일을 찾을 수 없습니다: {audio_file_for_json}")
                    logger.debug(f"JSON 파일 검색 실패: {audio_file_for_json} in {voice_dir}")
            else:
                logger.warning(f"    [경고] 음성 폴더를 찾을 수 없습니다: {voice_dir}")
                logger.debug(f"음성 폴더 없음: {voice_dir}")

            # 5-6. 중앙 마이크 WAV + JSON을 음성_output으로 복사
            logger.info(f"\n  [단계 6] 음성_output 폴더로 복사 중...")

            voice_output_wav = os.path.join(voice_output_dir, f"{base_name}.wav")

            if fallback_mode:
                # Fallback 모드: 음성 폴더 WAV를 동기화하여 음성_output에 저장
                logger.info(f"    → [Fallback] 음성 폴더 WAV 동기화 후 저장 중...")
                sync_audio_pair(wav_path, center_ch_file, voice_output_wav, -offset_samples, sr)
                logger.info(f"    → 동기화된 WAV 저장: {base_name}.wav")
                logger.debug(f"Fallback 동기화: {center_ch_file} → {voice_output_wav}")
            else:
                # 일반 모드: 중앙 마이크 WAV 파일 복사 (ch4 → 원본 이름으로)
                center_ch_output = os.path.join(output_dir, f"{base_name}_ch{config.CENTER_MIC_CHANNEL}.wav")

                if os.path.exists(center_ch_output):
                    shutil.copy2(center_ch_output, voice_output_wav)
                    logger.info(f"    → 중앙 마이크 WAV 복사: {base_name}.wav")
                    logger.debug(f"중앙 마이크 복사: {center_ch_output} → {voice_output_wav}")
                else:
                    logger.warning(f"    [경고] 중앙 마이크 파일을 찾을 수 없습니다: {center_ch_output}")
                    logger.debug(f"중앙 마이크 파일 없음: {center_ch_output}")

            # JSON 파일 복사
            if json_path and output_json_path and os.path.exists(output_json_path):
                voice_output_json = os.path.join(voice_output_dir, json_filename)
                shutil.copy2(output_json_path, voice_output_json)
                logger.info(f"    → JSON 파일 복사: {json_filename}")
                logger.debug(f"JSON 복사: {output_json_path} → {voice_output_json}")
            else:
                logger.warning(f"    [경고] JSON 파일을 찾을 수 없어 복사하지 않습니다.")
                logger.debug(f"JSON 파일 복사 불가: json_path={json_path}")

            file_elapsed_time = time.time() - file_start_time
            logger.info(f"  [완료] 음성_output 폴더로 복사 완료!")
            logger.info(f"  [완료] 파일 처리 완료 (소요 시간: {file_elapsed_time:.1f}초)\n")
            logger.debug(f"파일 처리 총 소요 시간: {file_elapsed_time:.1f}초")

            # 성공 카운트 증가
            success_count += 1

        except Exception as e:
            file_elapsed_time = time.time() - file_start_time
            error_msg = str(e)
            failed_files.append((aup3_file, error_msg))
            logger.error(f"  [오류] 오류 발생: {error_msg}")
            logger.error(f"  처리 시간: {file_elapsed_time:.1f}초")
            import traceback
            logger.debug(traceback.format_exc())
            continue

    logger.info("\n" + "=" * 70)
    logger.info("[완료] 동기화 완료!")
    logger.info(f"출력1: {os.path.basename(output_dir)} ({config.NUM_MIC_CHANNELS}채널 WAV+JSON)")
    logger.info(f"출력2: {os.path.basename(voice_output_dir)} (중앙ch{config.CENTER_MIC_CHANNEL} WAV+JSON)")
    logger.info("=" * 70)

    # 처리 결과 요약
    logger.info(f"\n[결과 요약]")
    logger.info(f"  전체: {len(matches)}개")
    logger.info(f"  성공: {success_count}개")
    logger.info(f"  실패: {len(failed_files)}개")

    # 실패 파일은 로그에만 기록 (failed_files.txt 생성 안 함)
    if failed_files:
        logger.error(f"\n[오류] 실패한 파일이 있습니다:")
        for filename, reason in failed_files:
            logger.error(f"  - {filename}: {reason}")

    # 임시 폴더 정리
    if os.path.exists(temp_base):
        logger.info("\n[정리] 임시 파일 삭제 중...")
        logger.debug(f"임시 폴더 삭제: {temp_base}")
        try:
            shutil.rmtree(temp_base)
            logger.info("[완료] 임시 파일 삭제 완료")
        except Exception as e:
            logger.warning(f"[경고] 임시 파일 삭제 실패: {e}")
            logger.debug(f"임시 파일 삭제 오류: {str(e)}")

    # 실패한 파일이 있으면 임시 파일로 기록 (batch_process에서 수집)
    if failed_files:
        failed_files_path = os.path.join(reference_dir, ".failed_files.tmp")
        try:
            with open(failed_files_path, 'w', encoding='utf-8') as f:
                for filename, reason in failed_files:
                    # 폴더 이름도 포함해서 기록
                    folder_name = os.path.basename(os.path.dirname(reference_dir))
                    f.write(f"{folder_name}\\{filename}\t{reason}\n")
            logger.debug(f"실패 파일 목록 저장: {failed_files_path}")
        except Exception as e:
            logger.warning(f"[경고] 실패 파일 목록 저장 실패: {e}")

    # 실패가 있으면 종료 코드 1 반환
    return 1 if failed_files else 0


if __name__ == "__main__":
    sys.exit(main())
