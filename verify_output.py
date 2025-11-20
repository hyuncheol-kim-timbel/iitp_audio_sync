#!/usr/bin/env python3
"""
생성된 WAV 파일들의 데이터 검증
"""

import soundfile as sf
import numpy as np
import os

output_dir = r'C:\Projects\IITP\20251113_자료\20250619\프로젝트_output'

# 첫 번째 파일 세트 확인 (001)
print('='*80)
print('출력 파일 검증 - VID_20250619_171147_00_001')
print('='*80)

for ch in range(1, 8):
    filename = f'VID_20250619_171147_00_001_ch{ch}.wav'
    filepath = os.path.join(output_dir, filename)

    if os.path.exists(filepath):
        audio, sr = sf.read(filepath)

        print(f'\nChannel {ch}: {filename}')
        print(f'  샘플레이트: {sr}Hz')
        print(f'  길이: {len(audio)} 샘플 ({len(audio)/sr:.2f}초)')
        print(f'  데이터 타입: {audio.dtype}')
        print(f'  통계:')
        print(f'    Min: {np.min(audio):.6f}')
        print(f'    Max: {np.max(audio):.6f}')
        print(f'    Mean: {np.mean(audio):.6f}')
        print(f'    Std: {np.std(audio):.6f}')

        # 값의 분포
        positive = np.sum(audio > 0)
        negative = np.sum(audio < 0)
        zero = np.sum(audio == 0)
        total = len(audio)

        print(f'  분포:')
        print(f'    양수: {positive} ({positive/total*100:.1f}%)')
        print(f'    음수: {negative} ({negative/total*100:.1f}%)')
        print(f'    0: {zero} ({zero/total*100:.1f}%)')

        # 처음 10개 샘플
        print(f'  처음 10개 샘플: {audio[:10]}')

        # 경고: 0 이하에만 값이 있는 경우
        if positive == 0 and negative > 0:
            print(f'  [경고] 모든 값이 0 이하입니다!')
        elif negative == 0 and positive > 0:
            print(f'  [경고] 모든 값이 0 이상입니다!')
        else:
            print(f'  [정상] 양수와 음수 값이 모두 존재합니다.')
    else:
        print(f'\n[오류] 파일을 찾을 수 없습니다: {filename}')

print('\n' + '='*80)
print('검증 완료')
print('='*80)
