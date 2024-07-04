from pydub import AudioSegment

def convert_ogg_to_wav(input_ogg_path, output_wav_path):
    # OGG 파일을 로드
    audio = AudioSegment.from_ogg(input_ogg_path)
    
    # WAV 파일로 변환하여 저장
    audio.export(output_wav_path, format="wav")

# 사용 예제
input_ogg_path = "/home/aicontest/DF/data/audio/test/TEST_00334.ogg"
output_wav_path = "./output.wav"
convert_ogg_to_wav(input_ogg_path, output_wav_path)
