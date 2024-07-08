from pydub import AudioSegment

def convert_ogg_to_wav(input_ogg_path, output_wav_path):
    # OGG 파일을 로드
    audio = AudioSegment.from_ogg(input_ogg_path)
    
    audio = audio.set_frame_rate(8000)
    # WAV 파일로 변환하여 저장
    audio.export(output_wav_path, format="wav")

# 사용 예제
# input_ogg_path = "/home/aicontest/DF/data/audio/test/TEST_49546.ogg"
input_ogg_path = "/home/aicontest/DF/data/audio/test/TEST_49956.ogg"
output_wav_path = "./TEST_49956.wav"
convert_ogg_to_wav(input_ogg_path, output_wav_path)
