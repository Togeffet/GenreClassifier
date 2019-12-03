import songread

audio_path = 'music/touch_the_sky.mp3'
x , sr = songread.read_song_file(audio_path)

print(type(x), type(sr))
print(x.shape, sr)