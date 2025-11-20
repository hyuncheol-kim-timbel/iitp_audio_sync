#!/usr/bin/env python3
"""
오디오 파일의 속성을 확인하는 스크립트
"""

import soundfile as sf
import os

def check_audio_info(file_path):
    """오디오 파일의 정보를 출력합니다."""
    info = sf.info(file_path)
    print(f"\n파일: {os.path.basename(file_path)}")
    print(f"  - 샘플링 레이트: {info.samplerate} Hz")
    print(f"  - 채널 수: {info.channels} ({'모노' if info.channels == 1 else '스테레오' if info.channels == 2 else f'{info.channels}채널'})")
    print(f"  - 포맷: {info.format}, {info.subtype}")
    print(f"  - 길이: {info.duration:.2f}초 ({info.frames} 프레임)")
    return info

if __name__ == "__main__":
    print("=" * 80)
    print("오디오 파일 속성 확인")
    print("=" * 80)

    # 음성 폴더 파일 확인
    print("\n[음성 폴더 - 기준]")
    audio_dir = "음성"
    if os.path.exists(audio_dir):
        wav_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
        for wav_file in wav_files[:2]:  # 처음 2개만 확인
            check_audio_info(os.path.join(audio_dir, wav_file))

    # 음성_360 폴더 파일 확인
    print("\n[음성_360 폴더 - 대상]")
    audio_dir = "음성_360"
    if os.path.exists(audio_dir):
        wav_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
        for wav_file in wav_files[:2]:  # 처음 2개만 확인
            check_audio_info(os.path.join(audio_dir, wav_file))

    print("\n" + "=" * 80)
