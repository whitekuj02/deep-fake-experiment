import numpy
import soundfile as sf
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# input can be a URL or a local path
input = './TEST_49956.wav'
separation = pipeline(
   Tasks.speech_separation,
   # model_revision='v1.0.2',
   # model_revision='v0.9.0',
   # model='damo/speech_separation_mossformer_8k_pytorch')
   model='damo/speech_mossformer2_separation_temporal_8k')
   # model='dengcunqin/speech_mossformer2_noise_reduction_16k')
result = separation(input)
for i, signal in enumerate(result['output_pcm_list']):
    save_file = f'./result/output_spk{i}.wav'
    sf.write(save_file, numpy.frombuffer(signal, dtype=numpy.int16), 8000)