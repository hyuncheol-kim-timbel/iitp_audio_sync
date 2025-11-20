#!/usr/bin/env python3
"""
오디오 동기화 메인 프로그램

두 개의 마이크로 녹음된 오디오 파일들을 동기화합니다.
JSON 파일을 포함한 폴더를 기준으로 다른 폴더의 오디오를 동기화합니다.
"""

__version__ = "1.0.1"

import os
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
    find_json_file
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

            if not channel_files:
                error_msg = "채널을 추출하지 못했습니다."
                logger.error(f"  [오류] {error_msg}")
                logger.debug(f"파일 처리 실패: {aup3_file} - {error_msg}")
                continue

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

            # 5-1-1. Trim 감지 및 적용 (음성 폴더의 WAV를 기준으로)
            logger.info(f"\n  [단계 1-1] Trim 범위 감지 중...")

            # 음성 폴더 경로 찾기
            parent_folder = os.path.dirname(reference_dir)
            voice_dir = os.path.join(parent_folder, "음성")

            trim_applied = False
            if os.path.exists(voice_dir):
                # .aup3 파일명에서 베이스 이름 추출 (확장자 제거)
                base_name = Path(aup3_file).stem
                reference_wav_name = base_name + ".wav"
                reference_wav_path = os.path.join(voice_dir, reference_wav_name)

                if os.path.exists(reference_wav_path):
                    logger.info(f"    → 기준 WAV 파일 발견: {reference_wav_name}")
                    logger.debug(f"기준 WAV 경로: {reference_wav_path}")

                    # 첫 번째 채널을 사용하여 trim 범위 감지
                    from sync_logic import find_trim_range_by_reference, apply_trim_to_files

                    trim_left, trim_right, sr_trim = find_trim_range_by_reference(
                        channel_files[0],
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
                        success_count = apply_trim_to_files(channel_files, trim_left, trim_right)

                        if success_count == config.NUM_MIC_CHANNELS:
                            logger.info(f"    → [완료] {success_count}개 채널 trim 완료")
                            logger.debug(f"Trim 적용 성공: {success_count}개 채널")
                            trim_applied = True
                        else:
                            logger.warning(f"    → [경고] {success_count}/{config.NUM_MIC_CHANNELS}개 채널만 trim됨")
                            logger.debug(f"Trim 부분 성공: {success_count}/{config.NUM_MIC_CHANNELS}")
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

            # 5-2. 중앙 마이크 채널 선택
            center_ch_file = channel_files[config.CENTER_MIC_CHANNEL - 1]  # 1-based → 0-based
            logger.info(f"\n  [단계 2] 중앙 마이크 채널 {config.CENTER_MIC_CHANNEL} 선택: {os.path.basename(center_ch_file)}")
            logger.debug(f"중앙 채널 파일: {center_ch_file}")

            # 5-3. 중앙 채널과 음성_360 비교하여 오프셋 계산
            logger.info(f"  [단계 3] 오프셋 계산 (중앙ch{config.CENTER_MIC_CHANNEL} ↔ 음성_360, {sync_duration}초)...")
            center_audio, sr = load_audio(center_ch_file, duration=sync_duration, sr=sr)
            target_audio, sr = load_audio(wav_path, duration=sync_duration, sr=sr)

            offset_seconds, offset_samples = calculate_time_offset(center_audio, target_audio, sr)
            logger.info(f"    → 오프셋: {offset_seconds:.3f}초 ({offset_samples} 샘플)")
            logger.debug(f"오프셋 계산 완료: offset_seconds={offset_seconds}, offset_samples={offset_samples}, sr={sr}")

            # 5-4. 모든 7채널에 동일한 오프셋 적용
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

            # 5-5. JSON 파일 처리 (음성 폴더에서 찾아서 프로젝트_output에 복사 및 타임스탬프 조정)
            logger.info(f"  [단계 5] JSON 파일 처리 중...")

            # 음성 폴더 경로 찾기
            # reference_dir의 부모 폴더에서 "음성" 폴더 찾기
            parent_folder = os.path.dirname(reference_dir)
            voice_dir = os.path.join(parent_folder, "음성")

            json_path = None
            if os.path.exists(voice_dir):
                # .aup3 파일명에서 베이스 이름 추출 (확장자 제거)
                base_name = Path(aup3_file).stem
                audio_file_for_json = base_name + ".wav"

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

            # 중앙 마이크 WAV 파일 복사 (ch4 → 원본 이름으로)
            center_ch_output = os.path.join(output_dir, f"{base_name}_ch{config.CENTER_MIC_CHANNEL}.wav")
            voice_output_wav = os.path.join(voice_output_dir, f"{base_name}.wav")

            if os.path.exists(center_ch_output):
                shutil.copy2(center_ch_output, voice_output_wav)
                logger.info(f"    → 중앙 마이크 WAV 복사: {base_name}.wav")
                logger.debug(f"중앙 마이크 복사: {center_ch_output} → {voice_output_wav}")
            else:
                logger.warning(f"    [경고] 중앙 마이크 파일을 찾을 수 없습니다: {center_ch_output}")
                logger.debug(f"중앙 마이크 파일 없음: {center_ch_output}")

            # JSON 파일 복사
            if json_path and os.path.exists(output_json_path):
                voice_output_json = os.path.join(voice_output_dir, json_filename)
                shutil.copy2(output_json_path, voice_output_json)
                logger.info(f"    → JSON 파일 복사: {json_filename}")
                logger.debug(f"JSON 복사: {output_json_path} → {voice_output_json}")
            else:
                logger.warning(f"    [경고] JSON 파일을 찾을 수 없어 복사하지 않습니다.")
                logger.debug(f"JSON 파일 복사 불가: json_path={json_path}, output_json_path exists={os.path.exists(output_json_path) if json_path else False}")

            file_elapsed_time = time.time() - file_start_time
            logger.info(f"  [완료] 음성_output 폴더로 복사 완료!")
            logger.info(f"  [완료] 파일 처리 완료 (소요 시간: {file_elapsed_time:.1f}초)\n")
            logger.debug(f"파일 처리 총 소요 시간: {file_elapsed_time:.1f}초")

        except Exception as e:
            file_elapsed_time = time.time() - file_start_time
            logger.error(f"  [오류] 오류 발생: {str(e)}")
            logger.error(f"  처리 시간: {file_elapsed_time:.1f}초")
            import traceback
            logger.debug(traceback.format_exc())
            continue

    logger.info("\n" + "=" * 70)
    logger.info("[완료] 동기화 완료!")
    logger.info(f"출력1: {os.path.basename(output_dir)} ({config.NUM_MIC_CHANNELS}채널 WAV+JSON)")
    logger.info(f"출력2: {os.path.basename(voice_output_dir)} (중앙ch{config.CENTER_MIC_CHANNEL} WAV+JSON)")
    logger.info("=" * 70)

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


if __name__ == "__main__":
    main()
