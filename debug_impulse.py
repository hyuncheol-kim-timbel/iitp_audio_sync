import numpy as np
import librosa
import sys

def analyze_impulses(audio_file, search_duration=10):
    """오디오 파일의 임펄스를 상세 분석"""
    print(f"\n분석 파일: {audio_file}")

    # 오디오 로드
    audio, sr = librosa.load(audio_file, sr=16000, mono=True)
    duration = len(audio) / sr
    print(f"전체 길이: {duration:.2f}초, 샘플링레이트: {sr}Hz")

    # 검색 범위 설정
    if search_duration:
        search_samples = int(search_duration * sr)
        search_audio = audio[:min(search_samples, len(audio))]
    else:
        search_audio = audio

    # 에너지 계산 (실제 sync_logic.py와 동일하게)
    window_size = int(0.01 * sr)  # 10ms
    hop_size = window_size // 2

    energy = np.array([
        np.sum(search_audio[i:i+window_size]**2)
        for i in range(0, len(search_audio) - window_size, hop_size)
    ])

    # 평균 및 표준편차
    mean_energy = np.mean(energy)
    std_energy = np.std(energy)
    print(f"평균 에너지: {mean_energy:.6f}")
    print(f"표준편차: {std_energy:.6f}")

    # 임계값 설정 (실제 코드: 평균 + 표준편차 x factor)
    threshold_factor = 3.0
    threshold = mean_energy + (std_energy * threshold_factor)
    print(f"임계값 (평균 + 표준편차×{threshold_factor}): {threshold:.6f}")

    # 임계값 초과 지점들
    impulse_indices = np.where(energy > threshold)[0]
    print(f"\n임계값 초과 프레임 개수: {len(impulse_indices)}")

    if len(impulse_indices) > 0:
        # 각 임펄스의 시간과 에너지 출력
        print("\n[임계값 초과 지점들 (상위 10개)]")
        print(f"{'시간(초)':<12} {'에너지':<15} {'상대비율'}")
        print("-" * 45)

        # 에너지 순으로 정렬
        sorted_indices = impulse_indices[np.argsort(energy[impulse_indices])[::-1]]

        for i, idx in enumerate(sorted_indices[:10]):
            time_pos = (idx * hop_size) / sr
            energy_val = energy[idx]
            ratio = (energy_val - mean_energy) / std_energy
            marker = " ← 최대" if i == 0 else ""
            print(f"{time_pos:8.3f}초    {energy_val:12.6f}    평균+{ratio:6.2f}σ{marker}")

        # 최대 에너지 임펄스
        max_energy_idx = impulse_indices[np.argmax(energy[impulse_indices])]
        max_impulse_time = (max_energy_idx * hop_size) / sr
        max_energy_val = energy[max_energy_idx]

        print(f"\n[현재 알고리즘이 선택하는 임펄스]")
        print(f"시간: {max_impulse_time:.3f}초")
        print(f"에너지: {max_energy_val:.6f} (평균+{(max_energy_val-mean_energy)/std_energy:.2f}σ)")
    else:
        # 임계값 초과 없음 -> fallback: 검색 구간 내 최댓값
        print("\n[경고] 임계값 초과 지점 없음 → 검색 구간 내 최댓값 사용")
        max_energy_idx = np.argmax(energy)
        max_impulse_time = (max_energy_idx * hop_size) / sr
        max_energy_val = energy[max_energy_idx]

        print(f"\n[현재 알고리즘이 선택하는 임펄스 (fallback)]")
        print(f"시간: {max_impulse_time:.3f}초")
        print(f"에너지: {max_energy_val:.6f} (평균+{(max_energy_val-mean_energy)/std_energy:.2f}σ)")

    # 특정 시간대의 에너지 확인
    print(f"\n[특정 시간대 에너지 확인]")
    for check_time in [2.5, 2.7, 2.9, 7.4, 7.5, 7.6, 9.7, 9.835, 10.0]:
        if check_time <= duration:
            frame_idx = int(check_time * sr / hop_size)
            if frame_idx < len(energy):
                e = energy[frame_idx]
                sigma_val = (e - mean_energy) / std_energy
                print(f"{check_time:6.3f}초: {e:12.6f} (평균+{sigma_val:6.2f}σ) {'← 임계값 초과' if e > threshold else ''}")

if __name__ == "__main__":
    audio_file = sys.argv[1] if len(sys.argv) > 1 else None
    search_duration = float(sys.argv[2]) if len(sys.argv) > 2 else 10

    if audio_file:
        analyze_impulses(audio_file, search_duration)
