import numpy as np
import audio

mel = np.load("test_mel.npy")
wav = audio.inv_mel_spectrogram(mel.T)
audio.save_wav(wav, "test_.wav")
