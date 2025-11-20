# 오디오 동기화 설정 파일

__version__ = "1.0.0"

# 동기화에 사용할 초기 오디오 길이 (초)
# 박수 소리(임펄스)가 이 구간 내에 있어야 합니다
SYNC_DURATION = 10

# 동기화 분석용 샘플링 레이트 (Hz)
# 출력 파일은 원본 파일의 샘플링 레이트를 유지합니다
# 높을수록 정확하지만 처리 속도가 느려집니다
SAMPLE_RATE = 16000

# 정밀 동기화 설정
# 업샘플링 배율 (높을수록 정확하지만 느림, 권장: 10)
UPSAMPLE_FACTOR = 10

# 임펄스 감지 임계값 계수 (평균 에너지의 몇 배, 권장: 3.0)
IMPULSE_THRESHOLD = 3.0

# 출력 폴더 접미사
OUTPUT_SUFFIX = "_output"

# 폴더 경로 설정
AUDIO_FOLDERS = {
    "reference": "프로젝트",    # 7채널 .aup3 파일 폴더
    "target": "음성_360"       # 동기화할 대상 폴더 (단일 채널)
}

# 파일 패턴
FILE_PATTERNS = {
    "reference": "VID_*.wav",
    "target": "LRV_*.wav"
}

# 동기화 방향 설정
# True: JSON이 없는 폴더(target)의 오디오를 조절 (기존 동작)
# False: JSON이 있는 폴더(reference)의 오디오를 조절하고 JSON timestamp도 조정
ADJUST_TARGET = False

# .aup3 다채널 처리 설정
# True: .aup3 파일의 모든 트랙을 개별 WAV 파일로 추출 (file_ch1.wav, file_ch2.wav, ...)
# False: 첫 번째 트랙만 추출
AUP3_EXTRACT_ALL_TRACKS = True

# 중앙 마이크 채널 번호 (1~7)
# 7채널 마이크 배치: 중앙 1개 + 주변 6개
# 중앙 마이크 채널을 음성_360 파일과 비교하여 오프셋 계산
# 계산된 오프셋은 모든 7개 채널에 동일하게 적용됨
CENTER_MIC_CHANNEL = 4  # 중앙 마이크가 4번 채널인 경우 (1~7 사이 값)

# 7채널 마이크 시스템 설정
NUM_MIC_CHANNELS = 7  # 총 마이크 채널 수