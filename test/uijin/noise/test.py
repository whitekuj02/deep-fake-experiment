from importlib import import_module
import sys
import torchaudio
import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import noisereduce as nr

# model_directory = "/home/aicontest/DF/test/uijin/noise/Audio-Denoising"
# sys.path.append(model_directory)
# module = import_module("denoise")
# AudioDeNoise = getattr(module, "AudioDeNoise")


# audioDenoiser = AudioDeNoise(inputFile="/home/aicontest/DF/data/audio/test/TEST_49898.ogg")
# audioDenoiser.deNoise(outputFile="input_denoised.wav")
# #audioDenoiser.generateNoiseProfile(noiseFile="input_noise_profile.wav")

audio_path = "/home/aicontest/DF/data/audio/test/TEST_49941.ogg"

audio, sr = librosa.load(audio_path, sr=32000)

# # normalized_audio = librosa.util.normalize(audio)

# # time = np.arange(0, len(audio)) / sr
# # plt.plot(time, audio)
# # plt.savefig("./result.png")

clean_audio = nr.reduce_noise(y=audio, sr=sr)

# # 오디오 파일 저장
sf.write('./result.wav', clean_audio, 32000)

# from speechbrain.inference.separation import SepformerSeparation as separator

# model = separator.from_hparams(source="speechbrain/sepformer-wsj02mix", savedir='pretrained_models/sepformer-wsj02mix')

# # for custom file, change path
# est_sources = model.separate_file(path='./result.wav')

# torchaudio.save("source1hat.wav", est_sources[:, :, 0].detach().cpu(), 8000)
# torchaudio.save("source2hat.wav", est_sources[:, :, 1].detach().cpu(), 8000)

from pydub import AudioSegment

sound1 = AudioSegment.from_ogg("/home/aicontest/DF/data/audio/train/AAACWKPZ.ogg")
sound2 = AudioSegment.from_ogg("/home/aicontest/DF/data/audio/train/AAAQOZYI.ogg")

# mix sound2 with sound1, starting at 5000ms into sound1)
output = sound1.overlay(sound2, position=0)

# save the result
output.export("./mixed_sounds.wav", format="wav")