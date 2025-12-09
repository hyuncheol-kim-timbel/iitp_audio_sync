"""
.sync 파일이 있는 폴더만 처리하는 테스트 스크립트
사용자가 수동으로 지정한 박수 위치(±0.5초)에서만 임펄스 검색
"""
import os
import sys
import glob
sys.path.insert(0, ".")

from aup3_converter import extract_all_tracks_from_aup3
from sync_logic import load_audio, detect_multiple_impulses
from scipy import signal
import numpy as np
import tempfile
import shutil

def parse_sync_file(sync_file_path):
    """
    .sync 파일에서 박수 위치를 읽습니다.
    형식: 00.000 (초 단위, 예: 04.526)

    Returns:
        float: 박수 위치 (초)
    """
    try:
        with open(sync_file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        # 00.000 형식 파싱
        time_sec = float(content)
        return time_sec
    except Exception as e:
        print(f"[오류] .sync 파일 읽기 실패: {e}")
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


def sync_with_manual_hint(ref_audio, target_audio, sr, ref_hint_time, target_hint_time=None):
    """
    수동 힌트를 사용하여 동기화

    Args:
        ref_audio: 기준 오디오
        target_audio: 대상 오디오
        sr: 샘플링 레이트
        ref_hint_time: 기준 오디오 힌트 시간 (초)
        target_hint_time: 대상 오디오 힌트 시간 (초, None이면 전체 검색)

    Returns:
        offset_seconds, correlation
    """
    # 기준 오디오에서 힌트 주변 임펄스 찾기
    ref_candidates = find_best_impulse_in_range(ref_audio, sr, ref_hint_time, search_range=0.5, top_n=5)

    # 대상 오디오에서 임펄스 찾기
    if target_hint_time is not None:
        target_candidates = find_best_impulse_in_range(target_audio, sr, target_hint_time, search_range=0.5, top_n=5)
    else:
        # 힌트 없으면 전체 10초 검색
        target_candidates = find_best_impulse_in_range(target_audio, sr, hint_time=5.0, search_range=5.0, top_n=10)

    print(f"    - 기준 후보: {len(ref_candidates)}개 (힌트: {ref_hint_time:.3f}초 ±0.5초)")
    print(f"    - 대상 후보: {len(target_candidates)}개")

    # 모든 조합의 correlation 계산
    best_correlation = 0
    best_ref_idx = 0
    best_target_idx = 0

    window_samples = int(1.0 * sr)  # 1초 윈도우

    for ref_idx, ref_energy in ref_candidates:
        ref_start = max(0, ref_idx - window_samples // 4)
        ref_end = min(len(ref_audio), ref_idx + window_samples)
        ref_segment = ref_audio[ref_start:ref_end]

        for target_idx, target_energy in target_candidates:
            target_start = max(0, target_idx - window_samples // 4)
            target_end = min(len(target_audio), target_idx + window_samples)
            target_segment = target_audio[target_start:target_end]

            if len(ref_segment) < sr * 0.1 or len(target_segment) < sr * 0.1:
                continue

            # Cross-correlation
            corr = signal.correlate(ref_segment, target_segment, mode="valid")
            ref_norm = np.sqrt(np.sum(ref_segment**2))
            target_norm = np.sqrt(np.sum(target_segment**2))

            if ref_norm > 0 and target_norm > 0:
                max_corr = np.max(corr) / (ref_norm * target_norm)
            else:
                max_corr = 0

            if max_corr > best_correlation:
                best_correlation = max_corr
                best_ref_idx = ref_idx
                best_target_idx = target_idx

    # 오프셋 계산
    offset_samples = best_ref_idx - best_target_idx
    offset_seconds = offset_samples / sr

    print(f"    - 최적 매칭: 기준 {best_ref_idx/sr:.3f}초, 대상 {best_target_idx/sr:.3f}초")
    print(f"    - Correlation: {best_correlation:.4f}")
    print(f"    - 오프셋: {offset_seconds:.3f}초")

    return offset_seconds, best_correlation


def process_folder_with_sync(folder_path):
    """
    .sync 파일이 있는 폴더 처리
    """
    project_folder = os.path.join(folder_path, "프로젝트")
    wav360_folder = os.path.join(folder_path, "음성_360")

    # .sync 파일 찾기
    sync_files = glob.glob(os.path.join(project_folder, "*.sync"))

    if not sync_files:
        print(f"[건너뜀] .sync 파일 없음: {folder_path}")
        return None

    print(f"\n{'='*80}")
    print(f"폴더: {os.path.basename(folder_path)}")
    print(f"{'='*80}")

    results = []

    for sync_file in sync_files:
        # 파일명 파싱
        base_name = os.path.basename(sync_file).replace(".sync", "")
        aup3_file = os.path.join(project_folder, f"{base_name}.aup3")

        # 매칭되는 360 WAV 파일 찾기
        parts = base_name.split("_")
        if len(parts) >= 5:
            wav360_name = f"LRV_{parts[1]}_{parts[2]}_11_{parts[4]}.wav"
            wav360_path = os.path.join(wav360_folder, wav360_name)
        else:
            print(f"[오류] 파일명 형식 불일치: {base_name}")
            continue

        if not os.path.exists(aup3_file):
            print(f"[오류] .aup3 파일 없음: {aup3_file}")
            continue

        if not os.path.exists(wav360_path):
            print(f"[오류] 360 WAV 파일 없음: {wav360_path}")
            continue

        # .sync 파일에서 힌트 읽기
        hint_time = parse_sync_file(sync_file)
        if hint_time is None:
            continue

        print(f"\n파일: {base_name}")
        print(f"  .sync 힌트: {hint_time:.3f}초")

        # 임시 폴더에 채널 추출
        temp_dir = tempfile.mkdtemp()
        try:
            # 7채널 추출
            print(f"  [1/3] 채널 추출 중...")
            tracks = extract_all_tracks_from_aup3(aup3_file, temp_dir)
            ch4_path = os.path.join(temp_dir, f"{base_name}_ch4.wav")

            if not os.path.exists(ch4_path):
                print(f"  [오류] ch4 추출 실패")
                continue

            # 오디오 로드
            print(f"  [2/3] 오디오 로딩...")
            ref_audio, sr = load_audio(ch4_path, duration=max(10, hint_time + 2), sr=16000)
            target_audio, sr = load_audio(wav360_path, duration=10, sr=16000)

            # 힌트 기반 동기화
            print(f"  [3/3] 동기화 처리 (힌트 ±0.5초 범위)...")
            offset, correlation = sync_with_manual_hint(ref_audio, target_audio, sr, hint_time)

            result = {
                "file": base_name,
                "hint": hint_time,
                "offset": offset,
                "correlation": correlation,
                "success": correlation > 0.3  # 낮은 임계값
            }
            results.append(result)

            status = "✓ 성공" if result["success"] else "✗ 실패"
            print(f"  {status}: offset={offset:.3f}초, corr={correlation:.4f}")

        finally:
            # 임시 폴더 삭제
            shutil.rmtree(temp_dir, ignore_errors=True)

    return results


# 메인 실행
if __name__ == "__main__":
    base_dir = r"C:\Projects\IITP\20251113_자료\test12"

    # .sync가 있는 폴더 찾기
    folders_with_sync = []
    for date_folder in sorted(glob.glob(os.path.join(base_dir, "202*"))):
        project_folder = os.path.join(date_folder, "프로젝트")
        sync_files = glob.glob(os.path.join(project_folder, "*.sync"))
        if sync_files:
            folders_with_sync.append(date_folder)

    print(f"{'='*80}")
    print(f".sync 파일이 있는 폴더: {len(folders_with_sync)}개")
    print(f"{'='*80}")

    all_results = []
    for folder in folders_with_sync:
        results = process_folder_with_sync(folder)
        if results:
            all_results.extend(results)

    # 최종 요약
    print(f"\n{'='*80}")
    print("테스트 결과 요약")
    print(f"{'='*80}")

    success_count = sum(1 for r in all_results if r["success"])
    total_count = len(all_results)

    print(f"총 처리: {total_count}개")
    print(f"성공: {success_count}개")
    print(f"실패: {total_count - success_count}개")

    if all_results:
        print(f"\n상세 결과:")
        for r in all_results:
            status = "✓" if r["success"] else "✗"
            print(f"  {status} {r['file'][:30]:<30} | hint={r['hint']:>6.3f}s | offset={r['offset']:>7.3f}s | corr={r['correlation']:.4f}")

    print(f"{'='*80}")
