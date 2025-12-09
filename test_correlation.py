"""실패/성공 파일의 correlation 비교 테스트"""
import sys
sys.path.insert(0, ".")
from sync_logic import load_audio, detect_multiple_impulses
from scipy import signal
import numpy as np

test_files = [
    # 실패 파일들
    ("F1_20250701_001",
     r"C:\Projects\IITP\20251113_자료\test12\20250701\프로젝트_output\VID_20250701_180651_00_001_ch4.wav",
     r"C:\Projects\IITP\20251113_자료\test12\20250701\음성_360\LRV_20250701_180651_11_001.wav"),
    ("F2_20250704_001",
     r"C:\Projects\IITP\20251113_자료\test12\20250704\프로젝트_output\VID_20250704_181246_00_001_ch4.wav",
     r"C:\Projects\IITP\20251113_자료\test12\20250704\음성_360\LRV_20250704_181246_11_001.wav"),
    ("F3_20250708_002",
     r"C:\Projects\IITP\20251113_자료\test12\20250708\프로젝트_output\VID_20250708_182027_00_002_ch4.wav",
     r"C:\Projects\IITP\20251113_자료\test12\20250708\음성_360\LRV_20250708_182027_11_002.wav"),
    ("F4_20250811_004",
     r"C:\Projects\IITP\20251113_자료\test12\20250811\프로젝트_output\VID_20250811_194214_00_004_ch4.wav",
     r"C:\Projects\IITP\20251113_자료\test12\20250811\음성_360\LRV_20250811_194214_11_004.wav"),
    ("F5_20250811_006",
     r"C:\Projects\IITP\20251113_자료\test12\20250811\프로젝트_output\VID_20250811_201520_00_006_ch4.wav",
     r"C:\Projects\IITP\20251113_자료\test12\20250811\음성_360\LRV_20250811_201520_11_006.wav"),
    # 성공 파일들
    ("S1_20250701_002",
     r"C:\Projects\IITP\20251113_자료\test12\20250701\프로젝트_output\VID_20250701_183518_00_002_ch4.wav",
     r"C:\Projects\IITP\20251113_자료\test12\20250701\음성_360\LRV_20250701_183518_11_002.wav"),
    ("S2_20250704_002",
     r"C:\Projects\IITP\20251113_자료\test12\20250704\프로젝트_output\VID_20250704_184623_00_002_ch4.wav",
     r"C:\Projects\IITP\20251113_자료\test12\20250704\음성_360\LRV_20250704_184623_11_002.wav"),
    ("S3_20250708_001",
     r"C:\Projects\IITP\20251113_자료\test12\20250708\프로젝트_output\VID_20250708_180803_00_001_ch4.wav",
     r"C:\Projects\IITP\20251113_자료\test12\20250708\음성_360\LRV_20250708_180803_11_001.wav"),
    ("S4_20250811_001",
     r"C:\Projects\IITP\20251113_자료\test12\20250811\프로젝트_output\VID_20250811_180910_00_001_ch4.wav",
     r"C:\Projects\IITP\20251113_자료\test12\20250811\음성_360\LRV_20250811_180910_11_001.wav"),
]

print("=" * 65)
print("Correlation Gap 분석: 실패(F) vs 성공(S) 파일 비교")
print("=" * 65)
print(f"{'File':<18} | {'Corr1':>6} | {'Corr2':>6} | {'Gap':>6} | Detect")
print("-" * 65)

for name, ch4, wav360 in test_files:
    try:
        ref_audio, sr = load_audio(ch4, duration=10, sr=16000)
        target_audio, sr = load_audio(wav360, duration=10, sr=16000)
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

        # 판정: gap < 0.1 이면 실패로 감지
        if gap < 0.1:
            verdict = "X FAIL"
        else:
            verdict = "O OK"

        print(f"{name:<18} | {corr1:>6.4f} | {corr2:>6.4f} | {gap:>6.4f} | {verdict}")
    except Exception as e:
        print(f"{name:<18} | ERR: {str(e)[:40]}")

print("=" * 65)
print("Gap < 0.1 인 경우 '잘못된 동기화'로 판정")
print("=" * 65)
