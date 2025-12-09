"""
오디오 동기화 핵심 로직
임펄스 감지 및 고정밀 동기화 알고리즘
"""

__version__ = "1.0.2"

import os
import json
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.interpolate import interp1d
from pathlib import Path
import config
import tempfile
from aup3_converter import extract_wav_from_aup3, extract_all_tracks_from_aup3


def parse_sync_file(sync_file_path):
    """
    .sync 파일에서 박수 위치를 읽습니다.
    형식: 00.000 (초 단위, 예: 04.526)

    Args:
        sync_file_path: .sync 파일 경로

    Returns:
        float: 박수 위치 (초), 오류 시 None
    """
    try:
        with open(sync_file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        # 특수 문자 마침표(․ U+2024)를 일반 마침표로 변환
        content = content.replace('․', '.')
        content = content.replace('\u2024', '.')
        # 00.000 형식 파싱
        time_sec = float(content)
        return time_sec
    except Exception as e:
        # 에러 메시지를 ASCII로 출력 (한글 인코딩 문제 방지)
        print(f"[ERROR] Failed to parse .sync file: {sync_file_path}")
        print(f"Content: {repr(content if 'content' in locals() else 'N/A')}")
        return None


def find_best_impulse_in_range(audio, sr, hint_time, search_range=0.5, top_n=5):
    """
    힌트 시간 주변 ±search_range 초 범위에서 가장 강한 임펄스를 찾습니다.

    Args:
        audio: 오디오 데이터
        sr: 샘플링 레이트
        hint_time: 힌트 시간 (초)
        search_range: 검색 범위 (±초)
        top_n: 상위 후보 개수

    Returns:
        list: [(샘플 인덱스, 에너지), ...]
    """
    # 검색 범위 계산
    start_time = max(0, hint_time - search_range)
    end_time = hint_time + search_range

    start_sample = int(start_time * sr)
    end_sample = min(int(end_time * sr), len(audio))

    # 검색 구간 추출
    search_audio = audio[start_sample:end_sample]

    # 임펄스 감지
    window_size = int(0.01 * sr)  # 10ms
    hop_size = window_size // 2

    energy = np.array([
        np.sum(search_audio[i:i+window_size]**2)
        for i in range(0, len(search_audio) - window_size, hop_size)
    ])

    # 상위 N개 에너지 피크 찾기
    peak_indices = []
    for _ in range(top_n):
        if len(energy) == 0:
            break
        max_idx = np.argmax(energy)
        max_energy = energy[max_idx]

        # 절대 샘플 인덱스로 변환
        sample_idx = start_sample + max_idx * hop_size
        peak_indices.append((sample_idx, max_energy))

        # 주변 제거 (중복 방지)
        exclude_range = int(0.1 * sr / hop_size)  # 100ms
        start_exclude = max(0, max_idx - exclude_range)
        end_exclude = min(len(energy), max_idx + exclude_range)
        energy[start_exclude:end_exclude] = 0

    return peak_indices


def load_audio(file_path, duration=None, sr=None):
    """
    오디오 파일을 로드합니다.
    .aup3 파일인 경우 자동으로 WAV로 변환 후 로드합니다.

    Args:
        file_path: 오디오 파일 경로 (.wav 또는 .aup3)
        duration: 로드할 길이 (초), None이면 전체 로드
        sr: 샘플링 레이트, None이면 원본 사용

    Returns:
        audio: 오디오 데이터
        sr: 샘플링 레이트
    """
    if sr is None:
        sr = config.SAMPLE_RATE

    # .aup3 파일인 경우 임시 WAV로 변환
    temp_wav = None
    if file_path.lower().endswith('.aup3'):
        print(f"  [변환] .aup3 파일 감지, WAV로 변환 중...")
        temp_wav = tempfile.mktemp(suffix='.wav')
        success = extract_wav_from_aup3(file_path, temp_wav)
        if not success:
            raise RuntimeError(f".aup3 파일 변환 실패: {file_path}")
        file_path = temp_wav

    try:
        audio, sr = librosa.load(file_path, sr=sr, duration=duration, mono=True)
        return audio, sr
    finally:
        # 임시 파일 정리
        if temp_wav and os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except:
                pass


def detect_impulse(audio, sr, threshold_factor=3.0, search_duration=None):
    """
    오디오에서 임펄스(박수 소리 등)를 감지합니다.
    지정된 검색 구간 내에서 가장 큰 에너지를 가진 임펄스를 찾습니다.

    Args:
        audio: 오디오 데이터
        sr: 샘플링 레이트
        threshold_factor: 임계값 계수 (평균 에너지의 몇 배)
        search_duration: 임펄스 검색 구간 (초), None이면 전체 오디오

    Returns:
        impulse_index: 임펄스가 감지된 샘플 인덱스
    """
    # 검색 구간 제한
    if search_duration is not None:
        search_samples = int(search_duration * sr)
        search_audio = audio[:min(search_samples, len(audio))]
    else:
        search_audio = audio

    # 짧은 윈도우로 에너지 계산 (10ms)
    window_size = int(0.01 * sr)
    hop_size = window_size // 2

    # 에너지 계산
    energy = np.array([
        np.sum(search_audio[i:i+window_size]**2)
        for i in range(0, len(search_audio) - window_size, hop_size)
    ])

    # 평균 및 표준편차 계산
    mean_energy = np.mean(energy)
    std_energy = np.std(energy)

    # 임계값: 평균 + (표준편차 * threshold_factor)
    threshold = mean_energy + (std_energy * threshold_factor)

    # 임계값을 초과하는 모든 지점 찾기
    impulse_indices = np.where(energy > threshold)[0]

    if len(impulse_indices) > 0:
        # 임계값을 초과한 지점들 중에서 가장 큰 에너지를 가진 지점 선택
        # (첫 번째가 아닌 최대 에너지 임펄스 찾기)
        max_energy_idx = impulse_indices[np.argmax(energy[impulse_indices])]
        impulse_index = max_energy_idx * hop_size
        return impulse_index
    else:
        # 임계값을 초과하는 임펄스가 없으면 검색 구간 내에서 최댓값 사용
        max_energy_idx = np.argmax(energy)
        impulse_index = max_energy_idx * hop_size
        return impulse_index


def detect_multiple_impulses(audio, sr, threshold_factor=3.0, search_duration=None, top_n=5):
    """
    오디오에서 상위 N개의 임펄스 후보를 감지합니다.
    카페 환경의 다양한 노이즈를 고려하여 여러 후보를 반환합니다.

    Args:
        audio: 오디오 데이터
        sr: 샘플링 레이트
        threshold_factor: 임계값 계수 (평균 에너지의 몇 배)
        search_duration: 임펄스 검색 구간 (초), None이면 전체 오디오
        top_n: 반환할 상위 후보 개수

    Returns:
        impulse_candidates: [(샘플 인덱스, 에너지 값), ...] 리스트 (에너지 높은 순)
    """
    # 검색 구간 제한
    if search_duration is not None:
        search_samples = int(search_duration * sr)
        search_audio = audio[:min(search_samples, len(audio))]
    else:
        search_audio = audio

    # 짧은 윈도우로 에너지 계산 (10ms)
    window_size = int(0.01 * sr)
    hop_size = window_size // 2

    # 에너지 계산
    energy = np.array([
        np.sum(search_audio[i:i+window_size]**2)
        for i in range(0, len(search_audio) - window_size, hop_size)
    ])

    # 평균 및 표준편차 계산
    mean_energy = np.mean(energy)
    std_energy = np.std(energy)

    # 임계값: 평균 + (표준편차 * threshold_factor)
    threshold = mean_energy + (std_energy * threshold_factor)

    # 임계값을 초과하는 모든 지점 찾기
    impulse_indices = np.where(energy > threshold)[0]

    if len(impulse_indices) == 0:
        # fallback: 상위 N개 에너지 피크 사용
        impulse_indices = np.arange(len(energy))

    # 에너지 순으로 정렬하여 상위 N개 선택
    sorted_indices = impulse_indices[np.argsort(energy[impulse_indices])[::-1]]
    top_indices = sorted_indices[:min(top_n, len(sorted_indices))]

    # 결과를 (샘플 인덱스, 에너지) 튜플 리스트로 반환
    candidates = [
        (idx * hop_size, energy[idx])
        for idx in top_indices
    ]

    return candidates


def find_best_impulse_pair(reference_audio, target_audio, sr,
                           ref_candidates, target_candidates,
                           window_duration=1.0):
    """
    임펄스 후보 쌍들 중에서 cross-correlation이 가장 높은 쌍을 찾습니다.
    동시 녹음된 파일은 박수 소리 구간의 파형이 유사하므로 높은 correlation을 보입니다.

    Args:
        reference_audio: 기준 오디오
        target_audio: 대상 오디오
        sr: 샘플링 레이트
        ref_candidates: 기준 오디오의 임펄스 후보 리스트 [(인덱스, 에너지), ...]
        target_candidates: 대상 오디오의 임펄스 후보 리스트 [(인덱스, 에너지), ...]
        window_duration: 비교할 구간 길이 (초)

    Returns:
        best_ref_impulse: 최적의 기준 임펄스 샘플 인덱스
        best_target_impulse: 최적의 대상 임펄스 샘플 인덱스
        best_correlation: 최고 correlation 값
    """
    window_samples = int(window_duration * sr)
    best_correlation = -np.inf
    best_ref_impulse = ref_candidates[0][0]
    best_target_impulse = target_candidates[0][0]

    print(f"    - 임펄스 후보 쌍 비교 중: {len(ref_candidates)} x {len(target_candidates)} = {len(ref_candidates) * len(target_candidates)}개")

    # 모든 후보 쌍의 correlation을 저장
    pair_scores = []

    for ref_idx, ref_energy in ref_candidates:
        # 기준 오디오에서 임펄스 주변 구간 추출
        ref_start = max(0, ref_idx - window_samples // 4)
        ref_end = min(len(reference_audio), ref_idx + window_samples)
        ref_segment = reference_audio[ref_start:ref_end]

        for target_idx, target_energy in target_candidates:
            # 대상 오디오에서 임펄스 주변 구간 추출
            target_start = max(0, target_idx - window_samples // 4)
            target_end = min(len(target_audio), target_idx + window_samples)
            target_segment = target_audio[target_start:target_end]

            # 짧은 구간이면 스킵
            if len(ref_segment) < sr * 0.1 or len(target_segment) < sr * 0.1:
                continue

            # Cross-correlation 계산 (정규화)
            correlation = signal.correlate(ref_segment, target_segment, mode='valid')

            # 정규화: 각 세그먼트의 에너지로 나눔
            ref_norm = np.sqrt(np.sum(ref_segment**2))
            target_norm = np.sqrt(np.sum(target_segment**2))

            if ref_norm > 0 and target_norm > 0:
                max_corr = np.max(correlation) / (ref_norm * target_norm)
            else:
                max_corr = 0

            # 결과 저장
            pair_scores.append((max_corr, ref_idx, target_idx))

            # 최고 점수 갱신
            if max_corr > best_correlation:
                best_correlation = max_corr
                best_ref_impulse = ref_idx
                best_target_impulse = target_idx

    # 상위 10개 결과 출력
    pair_scores.sort(reverse=True)
    print(f"    - [디버그] 상위 10개 correlation 쌍:")
    for i, (corr, ref_idx, tgt_idx) in enumerate(pair_scores[:10], 1):
        print(f"      {i}위: 기준 {ref_idx/sr:.2f}초 - 대상 {tgt_idx/sr:.2f}초 (correlation: {corr:.4f})")

    print(f"    - 최적 쌍: 기준 {best_ref_impulse/sr:.3f}초, 대상 {best_target_impulse/sr:.3f}초")
    print(f"    - Correlation 점수: {best_correlation:.4f}")

    # 동기화 신뢰도 검증
    warning_message = None

    # 1위-2위 차이 계산 (Gap) - 정보성 출력만
    if len(pair_scores) >= 2:
        gap = pair_scores[0][0] - pair_scores[1][0]
        print(f"    - 1위-2위 Gap: {gap:.4f}")

        # Gap 검증 제거: 박수를 거의 동시에 눌렀을 경우 차이가 거의 없을 수 있음
        # 사용자가 .sync 파일로 수동 확인하므로 Gap이 작아도 정상 처리

    # Correlation 절대값 검증 (사용자가 파형으로 직접 확인하므로 경고만 제거, 로그에 남기지 않음)
    # if best_correlation < 0.5 and warning_message is None:
    #     warning_message = f"낮은 correlation (corr={best_correlation:.4f} < 0.5)"
    #     print(f"    - [경고] ★★★ {warning_message} ★★★")

    return best_ref_impulse, best_target_impulse, best_correlation, warning_message


def calculate_time_offset_precise(reference_audio, target_audio, sr, ref_hint_time=None, target_hint_time=None, upsample_factor=10):
    """
    두 오디오 간의 시간 오프셋을 정밀하게 계산합니다.
    카페 환경의 다양한 노이즈를 고려하여 여러 임펄스 후보를 비교합니다.
    동시 녹음된 파일은 박수 구간의 파형이 유사하므로 correlation으로 실제 박수를 식별합니다.

    Args:
        reference_audio: 기준 오디오 (JSON 포함 폴더)
        target_audio: 대상 오디오 (동기화할 오디오)
        sr: 샘플링 레이트
        ref_hint_time: 기준 오디오의 힌트 시간 (초), None이면 전체 검색
        target_hint_time: 대상 오디오의 힌트 시간 (초), None이면 전체 검색
        upsample_factor: 업샘플링 배율 (정확도 향상)

    Returns:
        offset_seconds: 오프셋 (초)
        offset_samples: 오프셋 (원본 샘플 단위)
        sync_warning: 동기화 경고 메시지 (문제 없으면 None)
    """
    # 양쪽 모두 힌트가 있으면 target 힌트 위치를 기준으로 cross-correlation 사용
    # (Audacity 프로젝트에서 각 트랙의 offset이 다를 수 있으므로, ref 힌트 위치를 그대로 신뢰하지 않음)
    if ref_hint_time is not None and target_hint_time is not None:
        print(f"    - [이중 수동 힌트] 대상 {target_hint_time:.3f}초 기준, 기준 오디오에서 cross-correlation으로 검색")

        # target(음성_360)의 힌트 위치에서 임펄스 구간 추출
        target_impulse = int(target_hint_time * sr)
        window_samples = int(1.0 * sr)

        target_start = max(0, target_impulse - window_samples // 4)
        target_end = min(len(target_audio), target_impulse + window_samples)
        target_segment = target_audio[target_start:target_end]

        if len(target_segment) < sr * 0.1:
            print(f"    - [경고] 대상 오디오 구간이 너무 짧음, 자동 감지로 전환")
            ref_hint_time = None
            target_hint_time = None
        else:
            # reference(ch4) 전체에서 target_segment와 가장 유사한 위치 찾기
            # 검색 범위: 처음 30초 (박수는 보통 녹음 시작 부분에 있음)
            search_duration = 30.0
            search_samples = int(search_duration * sr)
            ref_search = reference_audio[:min(search_samples, len(reference_audio))]

            # Cross-correlation
            correlation = signal.correlate(ref_search, target_segment, mode='valid')
            max_corr_idx = np.argmax(correlation)
            max_corr_value = correlation[max_corr_idx]

            # 정규화된 correlation 값 계산
            target_energy = np.sqrt(np.sum(target_segment**2))
            if target_energy > 0:
                ref_window = ref_search[max_corr_idx:max_corr_idx + len(target_segment)]
                ref_energy = np.sqrt(np.sum(ref_window**2))
                if ref_energy > 0:
                    normalized_corr = max_corr_value / (target_energy * ref_energy)
                else:
                    normalized_corr = 0
            else:
                normalized_corr = 0

            # 실제 임펄스 위치 계산 (window_samples//4를 더해서 중앙 위치로 조정)
            ref_impulse = max_corr_idx + window_samples // 4

            print(f"    - [검색 결과] 기준 오디오에서 임펄스 발견: {ref_impulse/sr:.3f}초 (correlation: {normalized_corr:.4f})")
            print(f"    - [비교] 대상 힌트: {target_hint_time:.3f}초, 기준 힌트(원본): {ref_hint_time:.3f}초, 기준 발견: {ref_impulse/sr:.3f}초")

            sync_warning = None
            if normalized_corr < 0.3:
                sync_warning = f"낮은 correlation ({normalized_corr:.4f} < 0.3)"
                print(f"    - [경고] ★★★ {sync_warning} - 수동 확인 권장 ★★★")

            # 오프셋 계산 (기준에서 타겟을 빼면 타겟이 얼마나 앞서는지 계산됨)
            offset_samples_analysis = ref_impulse - target_impulse
            offset_seconds = offset_samples_analysis / sr

            print(f"    - 오프셋 계산: {offset_seconds:.3f}초 ({offset_samples_analysis} 샘플 @ {sr}Hz)")

            return offset_seconds, offset_samples_analysis, sync_warning
    elif ref_hint_time is not None:
        print(f"    - [수동 힌트] 기준 오디오 {ref_hint_time:.3f}초 ±0.5초 범위에서 검색...")
        ref_candidates = find_best_impulse_in_range(reference_audio, sr, ref_hint_time, search_range=0.5, top_n=5)
        target_candidates = detect_multiple_impulses(
            target_audio,
            sr,
            threshold_factor=config.IMPULSE_THRESHOLD,
            search_duration=config.IMPULSE_SEARCH_DURATION,
            top_n=config.IMPULSE_CANDIDATES
        )
    elif target_hint_time is not None:
        print(f"    - [수동 힌트] 대상 오디오 {target_hint_time:.3f}초 ±0.5초 범위에서 검색...")
        ref_candidates = detect_multiple_impulses(
            reference_audio,
            sr,
            threshold_factor=config.IMPULSE_THRESHOLD,
            search_duration=config.IMPULSE_SEARCH_DURATION,
            top_n=config.IMPULSE_CANDIDATES
        )
        target_candidates = find_best_impulse_in_range(target_audio, sr, target_hint_time, search_range=0.5, top_n=5)
    else:
        print(f"    - 임펄스 후보 감지 중 (검색 구간: {config.IMPULSE_SEARCH_DURATION}초)...")
        # 상위 N개 임펄스 후보 감지 (카페 노이즈 환경 대응)
        ref_candidates = detect_multiple_impulses(
            reference_audio,
            sr,
            threshold_factor=config.IMPULSE_THRESHOLD,
            search_duration=config.IMPULSE_SEARCH_DURATION,
            top_n=config.IMPULSE_CANDIDATES
        )
        target_candidates = detect_multiple_impulses(
            target_audio,
            sr,
            threshold_factor=config.IMPULSE_THRESHOLD,
            search_duration=config.IMPULSE_SEARCH_DURATION,
            top_n=config.IMPULSE_CANDIDATES
        )

    print(f"    - 기준 오디오 후보: {len(ref_candidates)}개 [{', '.join([f'{idx/sr:.2f}초' for idx, _ in ref_candidates[:3]])}...]")
    print(f"    - 대상 오디오 후보: {len(target_candidates)}개 [{', '.join([f'{idx/sr:.2f}초' for idx, _ in target_candidates[:3]])}...]")

    # 최적의 임펄스 쌍 찾기 (correlation 기반)
    ref_impulse, target_impulse, correlation_score, sync_warning = find_best_impulse_pair(
        reference_audio, target_audio, sr,
        ref_candidates, target_candidates,
        window_duration=1.0
    )

    # 동기화 신뢰도 확인 (사용자가 파형으로 직접 확인하므로 correlation 경고는 로그에 남기지 않음)
    # low_confidence = correlation_score < config.SYNC_CONFIDENCE_THRESHOLD
    # if low_confidence and sync_warning is None:
    #     sync_warning = f"낮은 correlation (corr={correlation_score:.4f} < {config.SYNC_CONFIDENCE_THRESHOLD})"
    #     print(f"    - [경고] ★★★ 동기화 신뢰도 낮음!...")

    # 임펄스 감지 여부 확인
    impulse_detected = (ref_impulse > 0 or target_impulse > 0)

    if impulse_detected:
        # 임펄스 기반 분석: 임펄스 주변 1초 구간만 추출 (계산 효율성)
        print(f"    - [모드] 임펄스 기반 고정밀 분석")
        window_samples = int(1.0 * sr)

        ref_start = max(0, ref_impulse - window_samples // 4)
        ref_end = min(len(reference_audio), ref_impulse + window_samples)
        ref_segment = reference_audio[ref_start:ref_end]

        target_start = max(0, target_impulse - window_samples // 4)
        target_end = min(len(target_audio), target_impulse + window_samples)
        target_segment = target_audio[target_start:target_end]
    else:
        # 임펄스 미감지: 로드된 전체 구간 분석 (SYNC_DURATION 이내)
        print(f"    - [경고] 임펄스가 감지되지 않았습니다!")
        print(f"    - [모드] 로드된 전체 {len(reference_audio)/sr:.1f}초 구간 분석 (정확도가 낮을 수 있음)")

        ref_start = 0
        ref_segment = reference_audio  # 이미 duration으로 제한됨 (최대 SYNC_DURATION초)

        target_start = 0
        target_segment = target_audio  # 이미 duration으로 제한됨 (최대 SYNC_DURATION초)

    print(f"    - 분석 구간: 기준 {len(ref_segment)/sr:.2f}초, 대상 {len(target_segment)/sr:.2f}초")

    # 업샘플링하여 정밀도 향상
    print(f"    - {upsample_factor}배 업샘플링 + Cross-correlation 계산 중...")
    # scipy의 resample 사용
    ref_upsampled = signal.resample(ref_segment, len(ref_segment) * upsample_factor)
    target_upsampled = signal.resample(target_segment, len(target_segment) * upsample_factor)

    # Cross-correlation 계산
    correlation = signal.correlate(ref_upsampled, target_upsampled, mode='full')

    # 최대 상관관계 지점 찾기
    max_corr_index = np.argmax(correlation)

    # 오프셋 계산 (업샘플링된 샘플 단위)
    offset_upsampled = max_corr_index - (len(target_upsampled) - 1)

    # 원본 샘플 단위로 변환
    offset_samples_fine = offset_upsampled / upsample_factor

    # 세그먼트 시작 위치 차이를 보정
    offset_samples_fine += (ref_start - target_start)

    # 초 단위로 변환
    offset_seconds = offset_samples_fine / sr

    # 신뢰도 낮은 경우 최종 결과에도 경고 표시
    if sync_warning:
        print(f"  ★ 주의: {sync_warning} - 수동 확인 권장 ★")

    return offset_seconds, int(round(offset_samples_fine)), sync_warning


def calculate_time_offset(reference_audio, target_audio, sr, ref_hint_time=None, target_hint_time=None):
    """
    두 오디오 간의 시간 오프셋을 계산합니다.
    Cross-correlation을 사용하여 동기화 지점을 찾습니다.

    Args:
        reference_audio: 기준 오디오 (JSON 포함 폴더)
        target_audio: 대상 오디오 (동기화할 오디오)
        sr: 샘플링 레이트
        ref_hint_time: 기준 오디오의 힌트 시간 (초), None이면 전체 검색
        target_hint_time: 대상 오디오의 힌트 시간 (초), None이면 전체 검색

    Returns:
        offset_seconds: 오프셋 (초)
        positive이면 target이 늦게 시작, negative이면 target이 일찍 시작
    """
    # 정밀 모드 사용 (임펄스 기반 + 업샘플링)
    return calculate_time_offset_precise(reference_audio, target_audio, sr, ref_hint_time=ref_hint_time, target_hint_time=target_hint_time, upsample_factor=10)


def sync_audio_pair(reference_path, target_path, output_path, offset_samples, analysis_sr):
    """
    오디오 쌍을 동기화하여 저장합니다.
    원본 파일의 샘플링 레이트, 양자화, 채널 수를 모두 유지합니다.
    .aup3 파일인 경우 자동으로 WAV로 변환 후 처리합니다.

    Args:
        reference_path: 기준 오디오 파일 경로 (수정하지 않음) (.wav 또는 .aup3)
        target_path: 대상 오디오 파일 경로 (.wav 또는 .aup3)
        output_path: 출력 파일 경로 (항상 .wav로 저장)
        offset_samples: 오프셋 (샘플 단위, analysis_sr 기준)
        analysis_sr: 분석에 사용한 샘플링 레이트
    """
    # .aup3 파일인 경우 임시 WAV로 변환
    temp_target_wav = None
    if target_path.lower().endswith('.aup3'):
        print(f"  → .aup3 파일 변환 중...")
        temp_target_wav = tempfile.mktemp(suffix='.wav')
        success = extract_wav_from_aup3(target_path, temp_target_wav)
        if not success:
            raise RuntimeError(f".aup3 파일 변환 실패: {target_path}")
        target_path = temp_target_wav

    try:
        # 원본 대상 파일의 정보 읽기
        original_info = sf.info(target_path)
        original_sr = original_info.samplerate
        original_subtype = original_info.subtype
        original_channels = original_info.channels

        # 전체 대상 오디오를 원본 샘플링 레이트로 로드
        print(f"      → 로딩 중 ({original_info.duration:.1f}초, {original_sr}Hz)...")
        target_audio, loaded_sr = sf.read(target_path, dtype='float32')

        # 스테레오인 경우 채널 차원 유지
        if len(target_audio.shape) == 1:
            # 모노: (samples,)
            is_mono = True
        else:
            # 스테레오 또는 다채널: (samples, channels)
            is_mono = False

        # 오프셋을 원본 샘플링 레이트로 변환
        # offset_samples는 analysis_sr 기준이므로 original_sr로 스케일링
        offset_samples_original = int(offset_samples * original_sr / analysis_sr)

        if offset_samples_original > 0:
            # target이 늦게 시작한 경우 (reference가 일찍 시작)
            # target의 앞부분에 무음 추가
            if is_mono:
                silence = np.zeros(offset_samples_original, dtype='float32')
                synced_audio = np.concatenate([silence, target_audio])
            else:
                silence = np.zeros((offset_samples_original, original_channels), dtype='float32')
                synced_audio = np.concatenate([silence, target_audio], axis=0)
            print(f"  → Target이 {offset_samples_original/original_sr:.3f}초 늦게 시작 -> 앞에 무음 추가")
        elif offset_samples_original < 0:
            # target이 일찍 시작한 경우 (reference가 늦게 시작)
            # target의 앞부분 제거
            samples_to_remove = abs(offset_samples_original)
            if samples_to_remove < len(target_audio):
                synced_audio = target_audio[samples_to_remove:]
                print(f"  → Target이 {samples_to_remove/original_sr:.3f}초 일찍 시작 -> 앞부분 제거")
            else:
                print(f"  [경고] 제거할 샘플 수가 오디오 길이보다 큽니다. 빈 오디오 생성")
                if is_mono:
                    synced_audio = np.array([], dtype='float32')
                else:
                    synced_audio = np.array([], dtype='float32').reshape(0, original_channels)
        else:
            # 완벽히 동기화됨
            synced_audio = target_audio
            print(f"  → 이미 동기화되어 있음")

        # 출력 디렉토리 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 동기화된 오디오를 원본 속성으로 저장
        sf.write(output_path, synced_audio, original_sr, subtype=original_subtype)
        print(f"      → 저장: {os.path.basename(output_path)}")

    finally:
        # 임시 파일 정리
        if temp_target_wav and os.path.exists(temp_target_wav):
            try:
                os.remove(temp_target_wav)
            except:
                pass


def get_index_from_filename(filename):
    """
    파일 이름에서 인덱스를 추출합니다.
    예: VID_20250619_171147_00_001.wav -> 001
        VID_20250619_171147_00_001.aup3 -> 001
        VID_20250619_171147_00_001_ch1.wav -> 001_ch1 (채널 정보 포함)
        VID_20250619_171147_00_001_ch2.wav -> 001_ch2
    """
    # 확장자 제거
    name_without_ext = filename.replace('.wav', '').replace('.aup3', '')

    # _ch{N} 패턴 확인
    import re
    channel_match = re.search(r'_ch(\d+)$', name_without_ext)

    if channel_match:
        # 채널 정보가 있는 경우
        channel_num = channel_match.group(0)  # _ch1, _ch2, ...
        name_without_channel = name_without_ext[:-len(channel_num)]

        # 인덱스 추출
        parts = name_without_channel.split('_')
        if len(parts) >= 5:
            index = parts[-1]
            return index + channel_num  # 예: "001_ch1"
    else:
        # 채널 정보가 없는 경우
        parts = name_without_ext.split('_')
        if len(parts) >= 5:
            index = parts[-1]
            return index  # 예: "001"

    return None


def find_matching_files(reference_dir, target_dir):
    """
    매칭되는 파일 쌍을 찾습니다.
    .wav 및 .aup3 파일을 모두 지원합니다.

    Returns:
        List of tuples: [(ref_file, target_file, index), ...]
    """
    # .wav 및 .aup3 파일 모두 찾기
    reference_files = sorted([f for f in os.listdir(reference_dir)
                             if f.endswith('.wav') or f.endswith('.aup3')])
    target_files = sorted([f for f in os.listdir(target_dir)
                          if f.endswith('.wav') or f.endswith('.aup3')])

    matches = []
    for ref_file in reference_files:
        ref_index = get_index_from_filename(ref_file)
        if ref_index:
            for target_file in target_files:
                target_index = get_index_from_filename(target_file)
                if target_index == ref_index:
                    matches.append((ref_file, target_file, ref_index))
                    break

    return matches


def find_aup3_and_wav_pairs(aup3_dir, wav_dir):
    """
    .aup3 파일과 대응하는 WAV 파일 쌍을 찾습니다.
    프로젝트 폴더(.aup3) + 음성_360 폴더(WAV) 매칭용

    Args:
        aup3_dir: .aup3 파일이 있는 디렉토리 (프로젝트 폴더)
        wav_dir: WAV 파일이 있는 디렉토리 (음성_360 폴더)

    Returns:
        List of tuples: [(aup3_file, wav_file, index), ...]
    """
    # .aup3 파일 찾기
    aup3_files = sorted([f for f in os.listdir(aup3_dir) if f.endswith('.aup3')])
    # WAV 파일 찾기
    wav_files = sorted([f for f in os.listdir(wav_dir) if f.endswith('.wav')])

    matches = []
    for aup3_file in aup3_files:
        # .aup3 파일에서 인덱스 추출
        aup3_index = get_index_from_filename(aup3_file)
        if aup3_index:
            # 채널 정보 제거 (있을 경우)
            import re
            aup3_index_base = re.sub(r'_ch\d+$', '', aup3_index)

            # 매칭되는 WAV 파일 찾기
            for wav_file in wav_files:
                wav_index = get_index_from_filename(wav_file)
                if wav_index:
                    wav_index_base = re.sub(r'_ch\d+$', '', wav_index)

                    if aup3_index_base == wav_index_base:
                        matches.append((aup3_file, wav_file, aup3_index_base))
                        break

    return matches


def adjust_json_timestamps(json_path, output_path, offset_seconds):
    """
    JSON 파일의 timestamp를 조정합니다.
    JSON을 파싱하여 timestamp 값을 직접 수정하고 원본 포맷으로 저장합니다.

    Args:
        json_path: 원본 JSON 파일 경로
        output_path: 출력 JSON 파일 경로
        offset_seconds: 조정할 오프셋 (초 단위)
                       양수: 무음이 추가된 경우 -> timestamp에 더하기
                       음수: 앞부분이 제거된 경우 -> timestamp에서 빼기
    """
    # offset을 소숫점 3자리까지만 반올림
    offset_seconds = round(offset_seconds, 3)
    print(f"  [디버그] JSON 조정 시작: offset={offset_seconds:.3f}초")

    # JSON 파일 읽기
    with open(json_path, 'r', encoding='utf-8') as f:
        original_text = f.read()  # 원본 포맷 확인용

    # JSON 데이터 파싱
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # data 배열의 각 항목에서 starttime, endtime 조정
    changed_count = 0
    if 'data' in data and isinstance(data['data'], list):
        for item in data['data']:
            # starttime 처리
            if 'starttime' in item and isinstance(item['starttime'], (int, float)):
                old_value = item['starttime']
                # offset을 적용한 후, 원래 값의 소숫점 자릿수 유지
                new_value = old_value + offset_seconds
                new_value = max(0, new_value)

                # 원래 값의 소숫점 자릿수 계산
                old_str = str(old_value)
                if '.' in old_str:
                    decimal_places = len(old_str.split('.')[1])
                    # 원래 자릿수로 반올림하여 정밀도 유지
                    new_value = round(new_value, decimal_places)

                if abs(new_value - old_value) > 0.001:
                    item['starttime'] = new_value
                    changed_count += 1
                    if changed_count <= 3:
                        print(f"  [디버그] starttime: {old_value}초 -> {new_value}초")

            # endtime 처리
            if 'endtime' in item and isinstance(item['endtime'], (int, float)):
                old_value = item['endtime']
                # offset을 적용한 후, 원래 값의 소숫점 자릿수 유지
                new_value = old_value + offset_seconds
                new_value = max(0, new_value)

                # 원래 값의 소숫점 자릿수 계산
                old_str = str(old_value)
                if '.' in old_str:
                    decimal_places = len(old_str.split('.')[1])
                    # 원래 자릿수로 반올림하여 정밀도 유지
                    new_value = round(new_value, decimal_places)

                if abs(new_value - old_value) > 0.001:
                    item['endtime'] = new_value
                    changed_count += 1

    # 조정된 JSON 저장
    # 원본이 한 줄 JSON인지 확인 (줄바꿈이 거의 없으면 한 줄 JSON)
    is_minified = original_text.count('\n') < 10

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        if is_minified:
            # 한 줄 JSON으로 저장 (원본 포맷 유지)
            json.dump(data, f, ensure_ascii=False, separators=(', ', ': '))
        else:
            # 들여쓰기 있는 JSON으로 저장
            json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"  [완료] JSON 타임스탬프 조정 완료: {output_path}")
    print(f"  → 오프셋: {offset_seconds:.3f}초 적용, {changed_count}개 timestamp 변경됨")


def find_json_file(audio_file, audio_dir):
    """
    오디오 파일에 대응하는 JSON 파일을 찾습니다.

    Args:
        audio_file: 오디오 파일명 (예: VID_20250619_171147_00_001.wav)
        audio_dir: 오디오 파일이 있는 디렉토리

    Returns:
        JSON 파일 경로 또는 None
    """
    # .wav를 제거하고 _sp.json을 붙여서 찾기
    base_name = audio_file.replace('.wav', '')
    json_name = f"{base_name}_sp.json"
    json_path = os.path.join(audio_dir, json_name)

    if os.path.exists(json_path):
        return json_path
    return None


def find_trim_range_by_reference(extracted_wav, reference_wav, downsample_factor=100):
    """
    교차상관을 사용하여 추출된 WAV와 reference WAV 사이의 오프셋을 찾습니다.
    Audacity에서 각 트랙의 시작 위치가 다를 수 있으므로, 이 오프셋을 감지합니다.

    Args:
        extracted_wav: .aup3에서 추출한 WAV 파일 경로
        reference_wav: 음성 폴더의 WAV 파일 경로 (정확한 타임라인 기준)
        downsample_factor: 다운샘플링 비율 (빠른 검색용)

    Returns:
        tuple: (trim_left_samples, trim_right_samples, original_sr)
               trim_left > 0: 추출된 오디오 앞부분 제거 필요
               trim_left < 0: 추출된 오디오 앞에 무음 추가 필요 (현재 0으로 반환)
               또는 실패시 (None, None, None)
    """
    try:
        # 파일 로드
        extracted_audio, sr1 = sf.read(extracted_wav)
        reference_audio, sr2 = sf.read(reference_wav)

        if sr1 != sr2:
            print(f"  [경고] 샘플레이트가 다릅니다: extracted={sr1}Hz, reference={sr2}Hz")
            return None, None, None

        sr = sr1

        # 모노로 변환
        if len(extracted_audio.shape) > 1:
            extracted_audio = np.mean(extracted_audio, axis=1)
        if len(reference_audio.shape) > 1:
            reference_audio = np.mean(reference_audio, axis=1)

        # 처음 60초만 사용하여 오프셋 감지 (속도 향상)
        max_duration = 60  # 초
        max_samples = int(max_duration * sr)
        ext_segment = extracted_audio[:min(max_samples, len(extracted_audio))]
        ref_segment = reference_audio[:min(max_samples, len(reference_audio))]

        # 다운샘플링 (빠른 검색)
        extracted_down = ext_segment[::downsample_factor]
        reference_down = ref_segment[::downsample_factor]

        # 교차상관 (mode='full' 사용 - 두 신호 길이가 같아도 동작)
        correlation = signal.correlate(extracted_down, reference_down, mode='full')
        max_idx = np.argmax(correlation)

        # 오프셋 계산: max_idx - (len(ref) - 1)이 0이면 정렬됨
        # 양수: reference가 extracted보다 뒤에 있음 → extracted 앞부분 제거 필요
        # 음수: reference가 extracted보다 앞에 있음 → extracted 앞에 무음 추가 필요
        offset_down = max_idx - (len(reference_down) - 1)
        offset_samples = offset_down * downsample_factor
        offset_seconds = offset_samples / sr

        print(f"    → 오프셋 감지: {offset_seconds:.3f}초 ({offset_samples} 샘플)")

        if offset_samples > 0:
            # 추출된 오디오 앞부분 제거 필요
            trim_left = offset_samples
            trim_right = len(extracted_audio) - (trim_left + len(reference_audio))
            trim_right = max(0, trim_right)  # 음수 방지
            print(f"    → 추출된 오디오 앞 {offset_seconds:.3f}초 제거 필요")
        elif offset_samples < 0:
            # 추출된 오디오 앞에 무음 추가 필요 (sync_audio_pair에서 처리됨)
            # 여기서는 trim_left=0으로 반환하고, 오프셋 정보만 출력
            trim_left = 0
            trim_right = 0
            print(f"    → 추출된 오디오가 {-offset_seconds:.3f}초 뒤에 시작 (sync에서 처리)")
        else:
            # 완벽히 정렬됨
            trim_left = 0
            trim_right = 0
            print(f"    → 이미 정렬되어 있음")

        return trim_left, trim_right, sr

    except Exception as e:
        print(f"  [오류] Trim 범위 찾기 실패: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def apply_trim_to_files(file_list, trim_left, trim_right):
    """
    여러 WAV 파일에 동일한 trim을 적용합니다 (인플레이스 수정).
    
    Args:
        file_list: trim할 WAV 파일 경로 리스트
        trim_left: 앞에서 제거할 샘플 수
        trim_right: 뒤에서 제거할 샘플 수
    
    Returns:
        int: 성공적으로 trim된 파일 수
    """
    success_count = 0
    
    for file_path in file_list:
        try:
            # 파일 로드
            audio, sr = sf.read(file_path)
            
            # Trim 적용
            if trim_right > 0:
                trimmed_audio = audio[trim_left:-trim_right]
            else:
                trimmed_audio = audio[trim_left:]
            
            # 덮어쓰기
            sf.write(file_path, trimmed_audio, sr)
            success_count += 1
            
        except Exception as e:
            print(f"  [경고] {os.path.basename(file_path)} trim 실패: {e}")
    
    return success_count
