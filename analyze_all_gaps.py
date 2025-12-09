"""test12 폴더의 전체 파일 Gap 분석"""
import sys
import os
import glob
sys.path.insert(0, ".")
from sync_logic import load_audio, detect_multiple_impulses
from scipy import signal
import numpy as np

# test12 폴더 경로
base_dir = r"C:\Projects\IITP\20251113_자료\test12"

# 5개 실패 파일 (수동 확인됨)
known_failures = {
    "VID_20250701_180651_00_001",
    "VID_20250704_181246_00_001",
    "VID_20250708_182027_00_002",
    "VID_20250811_194214_00_004",
    "VID_20250811_201520_00_006"
}

def find_all_file_pairs(base_dir):
    """모든 폴더에서 .aup3 파일과 매칭되는 WAV 파일 찾기"""
    pairs = []

    # 날짜 폴더들 순회
    for date_folder in sorted(glob.glob(os.path.join(base_dir, "202*"))):
        project_folder = os.path.join(date_folder, "프로젝트")
        wav360_folder = os.path.join(date_folder, "음성_360")

        if not os.path.exists(project_folder) or not os.path.exists(wav360_folder):
            continue

        # .aup3 파일들 찾기
        for aup3_file in glob.glob(os.path.join(project_folder, "*.aup3")):
            base_name = os.path.basename(aup3_file).replace(".aup3", "")

            # 매칭되는 360 WAV 파일 찾기
            # VID_20250701_180651_00_001 -> LRV_20250701_180651_11_001
            parts = base_name.split("_")
            if len(parts) >= 5:
                wav360_name = f"LRV_{parts[1]}_{parts[2]}_11_{parts[4]}.wav"
                wav360_path = os.path.join(wav360_folder, wav360_name)

                if os.path.exists(wav360_path):
                    # ch4 파일은 추출이 필요하므로 프로젝트_output에서 찾기
                    output_folder = os.path.join(date_folder, "프로젝트_output")
                    ch4_path = os.path.join(output_folder, f"{base_name}_ch4.wav")

                    if os.path.exists(ch4_path):
                        is_known_fail = base_name in known_failures
                        pairs.append((base_name, ch4_path, wav360_path, is_known_fail))

    return pairs

def calculate_gap(ch4_path, wav360_path):
    """Gap 계산"""
    try:
        ref_audio, sr = load_audio(ch4_path, duration=10, sr=16000)
        target_audio, sr = load_audio(wav360_path, duration=10, sr=16000)
        ref_cand = detect_multiple_impulses(ref_audio, sr, threshold_factor=3.0, search_duration=10, top_n=10)
        tgt_cand = detect_multiple_impulses(target_audio, sr, threshold_factor=3.0, search_duration=10, top_n=10)

        pair_scores = []
        window_samples = int(1.0 * sr)

        for ref_idx, ref_energy in ref_cand:
            ref_start = max(0, ref_idx - window_samples // 4)
            ref_end = min(len(ref_audio), ref_idx + window_samples)
            ref_segment = ref_audio[ref_start:ref_end]

            for target_idx, target_energy in tgt_cand:
                target_start = max(0, target_idx - window_samples // 4)
                target_end = min(len(target_audio), target_idx + window_samples)
                target_segment = target_audio[target_start:target_end]

                if len(ref_segment) < sr * 0.1 or len(target_segment) < sr * 0.1:
                    continue

                corr = signal.correlate(ref_segment, target_segment, mode="valid")
                ref_norm = np.sqrt(np.sum(ref_segment**2))
                target_norm = np.sqrt(np.sum(target_segment**2))

                if ref_norm > 0 and target_norm > 0:
                    max_corr = np.max(corr) / (ref_norm * target_norm)
                else:
                    max_corr = 0
                pair_scores.append(max_corr)

        pair_scores.sort(reverse=True)
        corr1 = pair_scores[0] if len(pair_scores) > 0 else 0
        corr2 = pair_scores[1] if len(pair_scores) > 1 else 0
        gap = corr1 - corr2

        return corr1, corr2, gap, None
    except Exception as e:
        return 0, 0, 0, str(e)

# 모든 파일 쌍 찾기
print("=" * 85)
print("test12 폴더 전체 파일 Gap 분석")
print("=" * 85)
print("\n파일 쌍 검색 중...")
pairs = find_all_file_pairs(base_dir)
print(f"총 {len(pairs)}개 파일 쌍 발견\n")

# Gap 계산
results = []
for i, (name, ch4, wav360, is_fail) in enumerate(pairs):
    print(f"[{i+1}/{len(pairs)}] {name[:30]:<30} 처리 중...", end="\r")
    corr1, corr2, gap, error = calculate_gap(ch4, wav360)
    results.append((name, corr1, corr2, gap, is_fail, error))

# 결과를 Gap 기준으로 정렬
results.sort(key=lambda x: x[3])  # gap 기준 오름차순

print("\n" + "=" * 85)
print(f"{'File':<32} | {'Corr1':>6} | {'Corr2':>6} | {'Gap':>7} | {'Status':<8}")
print("-" * 85)

for name, corr1, corr2, gap, is_fail, error in results:
    if error:
        print(f"{name[:32]:<32} | ERROR: {error[:50]}")
    else:
        status = "FAIL" if is_fail else "OK"
        marker = " <<<" if is_fail else ""
        print(f"{name[:32]:<32} | {corr1:>6.4f} | {corr2:>6.4f} | {gap:>7.4f} | {status:<8}{marker}")

print("=" * 85)

# Gap 분포 통계
gaps = [gap for _, _, _, gap, _, error in results if not error]
if gaps:
    print(f"\nGap 분포:")
    print(f"  최소값: {min(gaps):.4f}")
    print(f"  최대값: {max(gaps):.4f}")
    print(f"  평균값: {np.mean(gaps):.4f}")
    print(f"  중간값: {np.median(gaps):.4f}")

    # 5개 실패 파일의 Gap 범위
    fail_gaps = [gap for _, _, _, gap, is_fail, error in results if is_fail and not error]
    if fail_gaps:
        print(f"\n5개 실패 파일 Gap 범위:")
        print(f"  최소값: {min(fail_gaps):.4f}")
        print(f"  최대값: {max(fail_gaps):.4f}")
        print(f"\n권장 임계값: {max(fail_gaps) + 0.001:.4f} (실패 파일 최대값 + 0.001)")

print("=" * 85)
