import songread

audio_path = 'music/touch_the_sky.mp3'
zc, sc, sr, mf, chroma = songread.read_song_file(audio_path)

print("""Zero Crossings: {}\n
         Specral Centroids: {}\n\
         Spectral Rolloff:{}\n\
         MFCCS: {}\n\
         Chroma Freqs: {}\n""".format(zc, sc, sr, mf, chroma))