#!/usr/bin/env python3
"""
Audacity .aup3 파일에서 WAV 추출 모듈

Audacity 3.0 이상의 프로젝트 파일(.aup3)은 SQLite 데이터베이스입니다.
이 모듈은 .aup3 파일에서 오디오 데이터를 추출하여 WAV 파일로 변환합니다.
"""

__version__ = "1.0.2"

import sqlite3
import numpy as np
import soundfile as sf
import os
from pathlib import Path
import struct


def get_aup3_info(aup3_path):
    """
    .aup3 파일의 기본 정보를 추출합니다.

    Args:
        aup3_path: .aup3 파일 경로

    Returns:
        dict: {'sample_rate': int, 'num_tracks': int, 'track_info': list}
    """
    conn = sqlite3.connect(aup3_path)
    cursor = conn.cursor()

    try:
        # 테이블 목록 확인
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        info = {
            'sample_rate': 44100,  # 기본값
            'num_tracks': 0,
            'track_info': []
        }

        # autosave 테이블에서 프로젝트 정보 추출
        if 'autosave' in tables:
            cursor.execute("SELECT dict FROM autosave WHERE id='project'")
            row = cursor.fetchone()
            if row:
                # XML 형식의 프로젝트 정보 파싱
                project_xml = row[0]

                # 샘플레이트 추출
                import re
                rate_match = re.search(r'rate="(\d+)"', project_xml)
                if rate_match:
                    info['sample_rate'] = int(rate_match.group(1))

                # 트랙 개수 추출 (wavetrack 태그 개수)
                track_matches = re.findall(r'<wavetrack', project_xml)
                info['num_tracks'] = len(track_matches)

                # 각 트랙의 채널 정보 추출 (channel 속성)
                channel_matches = re.findall(r'channel="(\d+)"', project_xml)
                info['track_info'] = [{'channel': int(ch)} for ch in channel_matches]

        # sampleblocks 테이블에서 트랙별 블록 정보 추출
        if 'sampleblocks' in tables:
            # Audacity는 각 트랙의 블록을 tracknumber로 구분
            # 하지만 schema에 tracknumber가 없을 수 있으므로 blockid로 추정
            cursor.execute("PRAGMA table_info(sampleblocks)")
            columns = [col[1] for col in cursor.fetchall()]

            if 'tracknumber' in columns:
                # tracknumber 컬럼이 있는 경우
                cursor.execute("SELECT DISTINCT tracknumber FROM sampleblocks ORDER BY tracknumber")
                track_numbers = [row[0] for row in cursor.fetchall()]
                info['num_tracks'] = len(track_numbers)
            else:
                # tracknumber가 없는 경우, blockid 패턴으로 트랙 수 추정
                cursor.execute("SELECT COUNT(*) FROM sampleblocks")
                total_blocks = cursor.fetchone()[0]
                print(f"  [정보] {total_blocks}개의 오디오 블록 발견")

                # blockid % 8 패턴 확인 (8채널 검사)
                cursor.execute("SELECT blockid FROM sampleblocks ORDER BY blockid")
                all_blockids = [row[0] for row in cursor.fetchall()]

                if all_blockids:
                    min_blockid = min(all_blockids)

                    # 각 (blockid - min) % N의 분포 확인
                    for num_channels in [8, 7, 6, 5, 4, 3, 2]:
                        pattern_dist = {}
                        for bid in all_blockids:
                            mod = (bid - min_blockid) % num_channels
                            pattern_dist[mod] = pattern_dist.get(mod, 0) + 1

                        # 모든 나머지가 균등하게 분포되어 있는지 확인
                        if len(pattern_dist) == num_channels:
                            expected_per_channel = total_blocks // num_channels
                            # 오차 범위 ±1
                            if all(abs(count - expected_per_channel) <= 1 for count in pattern_dist.values()):
                                info['num_tracks'] = num_channels
                                print(f"  [감지] {num_channels}채널 패턴 감지 (채널당 약 {expected_per_channel}개 블록)")
                                break

                # 여전히 0이면 1로 설정
                if info['num_tracks'] == 0:
                    info['num_tracks'] = 1

        return info

    finally:
        conn.close()


def extract_all_tracks_from_aup3(aup3_path, output_dir, base_filename=None):
    """
    .aup3 파일의 모든 트랙을 개별 WAV 파일로 추출합니다.

    Args:
        aup3_path: Audacity 프로젝트 파일 (.aup3) 경로
        output_dir: 출력 디렉토리
        base_filename: 기본 파일명 (None이면 원본 파일명 사용)

    Returns:
        list: 생성된 WAV 파일 경로 리스트 (실패 시 빈 리스트)
    """
    print(f"  [변환] .aup3 다중 트랙 추출: {os.path.basename(aup3_path)}")

    if not os.path.exists(aup3_path):
        print(f"  [오류] 파일을 찾을 수 없습니다: {aup3_path}")
        return []

    # 기본 파일명 설정
    if base_filename is None:
        base_filename = Path(aup3_path).stem

    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    try:
        conn = sqlite3.connect(aup3_path)
        cursor = conn.cursor()

        # 트랙 정보 가져오기
        try:
            info = get_aup3_info(aup3_path)
        except sqlite3.DatabaseError as e:
            print(f"  [오류] ★★★ 파일 손상됨 - 수동 처리 필요 ★★★")
            print(f"  [오류] SQLite 오류: {e}")
            print(f"  [해결방법] Audacity에서 열어서 다른 이름으로 저장하거나 WAV로 내보내기")
            conn.close()
            return []
        sample_rate = info['sample_rate']
        num_tracks = info['num_tracks']

        print(f"  [정보] 샘플레이트: {sample_rate}Hz, 트랙 수: {num_tracks}")

        # 스키마 확인
        cursor.execute("PRAGMA table_info(sampleblocks)")
        columns = [col[1] for col in cursor.fetchall()]
        has_tracknumber = 'tracknumber' in columns

        output_files = []

        # blockid의 시작 값 찾기
        cursor.execute("SELECT MIN(blockid) FROM sampleblocks")
        min_blockid = cursor.fetchone()[0]

        print(f"  [정보] blockid 시작값: {min_blockid}")

        # 각 트랙 추출
        for track_idx in range(num_tracks):
            print(f"  [추출] 트랙 {track_idx + 1}/{num_tracks} 처리 중...")

            if has_tracknumber:
                # tracknumber 컬럼이 있는 경우
                cursor.execute("""
                    SELECT blockid, sampleformat, samples
                    FROM sampleblocks
                    WHERE tracknumber = ?
                    ORDER BY blockid
                """, (track_idx,))
            else:
                # tracknumber가 없으면 blockid의 패턴으로 채널 구분
                # (blockid - min_blockid) % num_tracks == track_idx인 블록 선택
                cursor.execute("""
                    SELECT blockid, sampleformat, samples
                    FROM sampleblocks
                    ORDER BY blockid
                """)

            blocks = cursor.fetchall()

            if not blocks:
                print(f"  [경고] 트랙 {track_idx}에 데이터가 없습니다.")
                continue

            # 트랙번호가 없는 경우, blockid 패턴으로 채널 분리
            if not has_tracknumber and num_tracks > 1:
                # (blockid - min_blockid) % num_tracks == track_idx인 블록만 선택
                blocks = [b for b in blocks if (b[0] - min_blockid) % num_tracks == track_idx]

            # 오디오 데이터 병합
            audio_data = []
            for block_num, block in enumerate(blocks, 1):
                blockid, sampleformat, samples = block[:3]

                if samples is None:
                    continue

                # BLOB을 numpy array로 변환
                # Audacity 형식 코드:
                #   131073 (0x20001) = int16Sample
                #   262144 (0x40000) = floatSample
                #   196608 (0x30000) = int24Sample
                # 일부 파일은 플래그가 추가되므로 비트 마스크로 검사

                # 문자열 형식 처리
                if isinstance(sampleformat, str):
                    if sampleformat == 'int16Sample':
                        samples_array = np.frombuffer(samples, dtype='<i2').astype(np.float32) / 32768.0
                    elif sampleformat == 'floatSample':
                        samples_array = np.frombuffer(samples, dtype='<f4')
                    elif sampleformat == 'int24Sample':
                        print(f"  [경고] int24 형식은 현재 지원하지 않습니다. 블록 건너뜀")
                        continue
                    else:
                        print(f"  [경고] 알 수 없는 문자열 형식 ({sampleformat}), int16으로 시도")
                        try:
                            samples_array = np.frombuffer(samples, dtype='<i2').astype(np.float32) / 32768.0
                        except:
                            continue
                elif sampleformat is None or (isinstance(sampleformat, int) and sampleformat & 0x40000):
                    # float32 형식: 0x40000 비트가 설정됨 (262144, 262159 등)
                    samples_array = np.frombuffer(samples, dtype='<f4')
                elif isinstance(sampleformat, int) and sampleformat & 0x20000:
                    # int16 형식: 0x20000 비트가 설정됨 (131073 등)
                    samples_array = np.frombuffer(samples, dtype='<i2').astype(np.float32) / 32768.0
                elif isinstance(sampleformat, int) and sampleformat & 0x30000:
                    # int24 형식: 0x30000 비트가 설정됨 (196608 등)
                    print(f"  [경고] int24 형식은 현재 지원하지 않습니다. 블록 건너뜀")
                    continue
                else:
                    # 알 수 없는 형식 - int16으로 시도
                    print(f"  [경고] 알 수 없는 형식 ({sampleformat}), int16으로 시도")
                    try:
                        samples_array = np.frombuffer(samples, dtype='<i2').astype(np.float32) / 32768.0
                    except:
                        continue

                audio_data.append(samples_array)

            if not audio_data:
                print(f"  [경고] 트랙 {track_idx}에 유효한 오디오 데이터가 없습니다.")
                continue

            # 모든 블록 합치기
            full_audio = np.concatenate(audio_data)

            # 출력 파일명: {base_filename}_ch{N}.wav
            output_filename = f"{base_filename}_ch{track_idx + 1}.wav"
            output_path = os.path.join(output_dir, output_filename)

            # WAV 파일로 저장
            sf.write(output_path, full_audio, sample_rate, subtype='PCM_16')
            output_files.append(output_path)

            print(f"  [완료] ch{track_idx + 1}: {output_filename} ({len(full_audio) / sample_rate:.2f}초)")

        conn.close()

        print(f"  [완료] 총 {len(output_files)}개 트랙 추출 완료")
        return output_files

    except Exception as e:
        print(f"  [오류] 변환 실패: {e}")
        import traceback
        traceback.print_exc()
        return []


def extract_wav_from_aup3(aup3_path, output_wav_path, track_index=0):
    """
    .aup3 파일에서 WAV 파일을 추출합니다. (단일 트랙)

    Args:
        aup3_path: Audacity 프로젝트 파일 (.aup3) 경로
        output_wav_path: 출력 WAV 파일 경로
        track_index: 추출할 트랙 인덱스 (0부터 시작, 기본값: 0 - 첫 번째 트랙)

    Returns:
        bool: 성공 여부
    """
    print(f"  [변환] .aup3 파일 분석 중: {os.path.basename(aup3_path)}")

    if not os.path.exists(aup3_path):
        print(f"  [오류] 파일을 찾을 수 없습니다: {aup3_path}")
        return False

    try:
        conn = sqlite3.connect(aup3_path)
        cursor = conn.cursor()

        # 1. 프로젝트 정보 추출
        info = get_aup3_info(aup3_path)
        sample_rate = info['sample_rate']
        print(f"  [정보] 샘플레이트: {sample_rate}Hz")

        # 2. sampleblocks 테이블에서 오디오 데이터 추출
        cursor.execute("""
            SELECT blockid, sampleformat, summin, summax, sumrms, summary256, summary64k, samples
            FROM sampleblocks
            ORDER BY blockid
        """)

        blocks = cursor.fetchall()

        if not blocks:
            print(f"  [오류] 오디오 데이터를 찾을 수 없습니다.")
            conn.close()
            return False

        print(f"  [정보] {len(blocks)}개의 오디오 블록 추출 중...")

        # 3. 오디오 데이터 병합
        audio_data = []

        for block in blocks:
            blockid, sampleformat, summin, summax, sumrms, summary256, summary64k, samples = block

            if samples is None:
                continue

            # samples는 BLOB 형식으로 저장되어 있음
            # sampleformat에 따라 디코딩 방식이 다름
            # 일반적으로 'floatSample'은 32비트 float

            # BLOB을 numpy array로 변환
            # Audacity 형식 코드:
            #   131073 (0x20001) = int16Sample
            #   262144 (0x40000) = floatSample
            #   196608 (0x30000) = int24Sample
            # 일부 파일은 플래그가 추가되므로 비트 마스크로 검사

            # 문자열 형식 처리
            if isinstance(sampleformat, str):
                if sampleformat == 'int16Sample':
                    samples_array = np.frombuffer(samples, dtype='<i2').astype(np.float32) / 32768.0
                elif sampleformat == 'floatSample':
                    samples_array = np.frombuffer(samples, dtype='<f4')
                elif sampleformat == 'int24Sample':
                    print(f"  [경고] 블록 {blockid}: int24 형식은 현재 지원하지 않습니다. 건너뜀")
                    continue
                else:
                    print(f"  [경고] 블록 {blockid}: 알 수 없는 문자열 형식 ({sampleformat}), int16으로 시도")
                    try:
                        samples_array = np.frombuffer(samples, dtype='<i2').astype(np.float32) / 32768.0
                    except:
                        print(f"  [경고] 블록 {blockid} 변환 실패, 건너뜀")
                        continue
            elif sampleformat is None or (isinstance(sampleformat, int) and sampleformat & 0x40000):
                # float32 형식: 0x40000 비트가 설정됨 (262144, 262159 등)
                samples_array = np.frombuffer(samples, dtype='<f4')
            elif isinstance(sampleformat, int) and sampleformat & 0x20000:
                # int16 형식: 0x20000 비트가 설정됨 (131073 등)
                samples_array = np.frombuffer(samples, dtype='<i2').astype(np.float32) / 32768.0
            elif isinstance(sampleformat, int) and sampleformat & 0x30000:
                # int24 형식: 0x30000 비트가 설정됨 (196608 등)
                print(f"  [경고] 블록 {blockid}: int24 형식은 현재 지원하지 않습니다. 건너뜀")
                continue
            else:
                # 알 수 없는 형식 - int16으로 시도
                print(f"  [경고] 블록 {blockid}: 알 수 없는 형식 ({sampleformat}), int16으로 시도")
                try:
                    samples_array = np.frombuffer(samples, dtype='<i2').astype(np.float32) / 32768.0
                except:
                    print(f"  [경고] 블록 {blockid} 변환 실패, 건너뜀")
                    continue

            audio_data.append(samples_array)

        # 4. 모든 블록 합치기
        if not audio_data:
            print(f"  [오류] 추출된 오디오 데이터가 없습니다.")
            conn.close()
            return False

        full_audio = np.concatenate(audio_data)

        print(f"  [정보] 총 길이: {len(full_audio) / sample_rate:.2f}초")
        print(f"  [정보] 샘플 수: {len(full_audio)}")

        # 5. WAV 파일로 저장
        os.makedirs(os.path.dirname(output_wav_path) or '.', exist_ok=True)
        sf.write(output_wav_path, full_audio, sample_rate, subtype='PCM_16')

        print(f"  [완료] WAV 추출 완료: {output_wav_path}")

        conn.close()
        return True

    except sqlite3.Error as e:
        print(f"  [오류] SQLite 오류: {e}")
        return False
    except Exception as e:
        print(f"  [오류] 변환 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def convert_aup3_to_wav(aup3_path, output_dir=None, keep_original_name=False):
    """
    .aup3 파일을 WAV로 변환합니다.

    Args:
        aup3_path: .aup3 파일 경로
        output_dir: 출력 디렉토리 (None이면 원본 파일과 같은 위치)
        keep_original_name: True이면 원본 파일명 유지, False이면 .wav 확장자로 변경

    Returns:
        str: 생성된 WAV 파일 경로 (실패 시 None)
    """
    aup3_path = Path(aup3_path)

    if not aup3_path.exists():
        print(f"[오류] 파일을 찾을 수 없습니다: {aup3_path}")
        return None

    # 출력 파일명 결정
    if output_dir is None:
        output_dir = aup3_path.parent
    else:
        output_dir = Path(output_dir)

    if keep_original_name:
        output_name = aup3_path.stem + '.wav'
    else:
        output_name = aup3_path.stem + '.wav'

    output_path = output_dir / output_name

    # 변환 수행
    success = extract_wav_from_aup3(str(aup3_path), str(output_path))

    return str(output_path) if success else None


def batch_convert_aup3(input_dir, output_dir=None, recursive=False):
    """
    디렉토리 내의 모든 .aup3 파일을 WAV로 변환합니다.

    Args:
        input_dir: 입력 디렉토리
        output_dir: 출력 디렉토리 (None이면 입력 디렉토리와 동일)
        recursive: 하위 디렉토리까지 탐색할지 여부

    Returns:
        list: 생성된 WAV 파일 경로 리스트
    """
    input_dir = Path(input_dir)

    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # .aup3 파일 찾기
    if recursive:
        aup3_files = list(input_dir.rglob('*.aup3'))
    else:
        aup3_files = list(input_dir.glob('*.aup3'))

    if not aup3_files:
        print(f"[정보] .aup3 파일을 찾을 수 없습니다: {input_dir}")
        return []

    print(f"[발견] {len(aup3_files)}개의 .aup3 파일 발견")
    print("=" * 80)

    converted_files = []

    for i, aup3_file in enumerate(aup3_files, 1):
        print(f"\n[{i}/{len(aup3_files)}] 변환 중: {aup3_file.name}")

        # 상대 경로 유지 (recursive인 경우)
        if recursive:
            relative_path = aup3_file.relative_to(input_dir)
            output_subdir = output_dir / relative_path.parent
            output_subdir.mkdir(parents=True, exist_ok=True)
        else:
            output_subdir = output_dir

        wav_path = convert_aup3_to_wav(aup3_file, output_subdir)

        if wav_path:
            converted_files.append(wav_path)

    print("\n" + "=" * 80)
    print(f"[완료] {len(converted_files)}/{len(aup3_files)}개 파일 변환 성공")

    return converted_files


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Audacity .aup3 파일을 WAV로 변환',
        epilog='예시: python aup3_converter.py input.aup3 output.wav'
    )
    parser.add_argument('input', help='.aup3 파일 또는 디렉토리')
    parser.add_argument('output', nargs='?', default=None, help='출력 WAV 파일 또는 디렉토리')
    parser.add_argument('--batch', action='store_true', help='디렉토리 내 모든 .aup3 파일 변환')
    parser.add_argument('--recursive', action='store_true', help='하위 디렉토리까지 탐색 (--batch와 함께 사용)')
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')

    args = parser.parse_args()

    if args.batch:
        # 배치 변환
        batch_convert_aup3(args.input, args.output, args.recursive)
    else:
        # 단일 파일 변환
        if os.path.isdir(args.input):
            print("[오류] 디렉토리를 지정했습니다. --batch 옵션을 사용하세요.")
        else:
            result = convert_aup3_to_wav(args.input, args.output)
            if result:
                print(f"\n[성공] 변환 완료: {result}")
            else:
                print("\n[실패] 변환 실패")
