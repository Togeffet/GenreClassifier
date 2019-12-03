import librosa
audio_path = 'music/touch_the_sky.mp3'
x , sr = librosa.load(audio_path)
print(type(x), type(sr))

print(x.shape, sr)
