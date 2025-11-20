"""
오디오 동기화 핵심 로직
임펄스 감지 및 고정밀 동기화 알고리즘
"""

__version__ = "1.0.0"

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


def detect_impulse(audio, sr, threshold_factor=3.0):
    """
    오디오에서 임펄스(박수 소리 등)를 감지합니다.

    Args:
        audio: 오디오 데이터
        sr: 샘플링 레이트
        threshold_factor: 임계값 계수 (평균 에너지의 몇 배)

    Returns:
        impulse_index: 임펄스가 감지된 샘플 인덱스
    """
    # 짧은 윈도우로 에너지 계산 (10ms)
    window_size = int(0.01 * sr)
    hop_size = window_size // 2

    # 에너지 계산
    energy = np.array([
        np.sum(audio[i:i+window_size]**2)
        for i in range(0, len(audio) - window_size, hop_size)
    ])

    # 평균 및 표준편차 계산
    mean_energy = np.mean(energy)
    std_energy = np.std(energy)

    # 임계값: 평균 + (표준편차 * threshold_factor)
    threshold = mean_energy + (std_energy * threshold_factor)

    # 임계값을 초과하는 첫 번째 지점 찾기
    impulse_indices = np.where(energy > threshold)[0]

    if len(impulse_indices) > 0:
        # 첫 번째 임펄스의 실제 샘플 인덱스로 변환
        impulse_index = impulse_indices[0] * hop_size
        return impulse_index
    else:
        # 임펄스를 찾지 못한 경우 0 반환
        return 0


def calculate_time_offset_precise(reference_audio, target_audio, sr, upsample_factor=10):
    """
    두 오디오 간의 시간 오프셋을 정밀하게 계산합니다.
    업샘플링을 통해 서브샘플 정확도를 달성합니다.

    Args:
        reference_audio: 기준 오디오 (JSON 포함 폴더)
        target_audio: 대상 오디오 (동기화할 오디오)
        sr: 샘플링 레이트
        upsample_factor: 업샘플링 배율 (정확도 향상)

    Returns:
        offset_seconds: 오프셋 (초)
        offset_samples: 오프셋 (원본 샘플 단위)
    """
    print(f"    - 임펄스 감지 중...")

    # 임펄스 감지
    ref_impulse = detect_impulse(reference_audio, sr, threshold_factor=config.IMPULSE_THRESHOLD)
    target_impulse = detect_impulse(target_audio, sr, threshold_factor=config.IMPULSE_THRESHOLD)

    print(f"    - 기준 임펄스 위치: {ref_impulse/sr:.3f}초")
    print(f"    - 대상 임펄스 위치: {target_impulse/sr:.3f}초")

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

    return offset_seconds, int(round(offset_samples_fine))


def calculate_time_offset(reference_audio, target_audio, sr):
    """
    두 오디오 간의 시간 오프셋을 계산합니다.
    Cross-correlation을 사용하여 동기화 지점을 찾습니다.

    Args:
        reference_audio: 기준 오디오 (JSON 포함 폴더)
        target_audio: 대상 오디오 (동기화할 오디오)
        sr: 샘플링 레이트

    Returns:
        offset_seconds: 오프셋 (초)
        positive이면 target이 늦게 시작, negative이면 target이 일찍 시작
    """
    # 정밀 모드 사용 (임펄스 기반 + 업샘플링)
    return calculate_time_offset_precise(reference_audio, target_audio, sr, upsample_factor=10)


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
    교차상관을 사용하여 추출된 WAV에서 reference WAV가 위치한 범위를 찾습니다.
    (Audacity의 trimLeft/trimRight 자동 감지용)
    
    Args:
        extracted_wav: .aup3에서 추출한 전체 WAV 파일 경로
        reference_wav: 음성 폴더의 trim된 WAV 파일 경로 (정답)
        downsample_factor: 다운샘플링 비율 (빠른 검색용)
    
    Returns:
        tuple: (trim_left_samples, trim_right_samples, original_sr)
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
        
        # 다운샘플링 (빠른 검색)
        extracted_down = extracted_audio[::downsample_factor]
        reference_down = reference_audio[::downsample_factor]
        
        # 교차상관
        correlation = signal.correlate(extracted_down, reference_down, mode='valid')
        max_idx = np.argmax(correlation)
        
        # 다운샘플 인덱스를 원본 인덱스로 변환
        trim_left = max_idx * downsample_factor
        trim_right = len(extracted_audio) - (trim_left + len(reference_audio))
        
        return trim_left, trim_right, sr
        
    except Exception as e:
        print(f"  [오류] Trim 범위 찾기 실패: {e}")
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
