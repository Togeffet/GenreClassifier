import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import songread
from gradientdescent import gradient_descent

#audio_path = 'music/touch_the_sky.mp3'
#zc, sc, sr, mf, chroma = songread.read_song_file(audio_path)

#print("""Zero Crossings: {}\n
#         Specral Centroids: {}\n\
#         Spectral Rolloff:{}\n\
#         MFCCS: {}\n\
#         Chroma Freqs: {}\n""".format(zc, sc, sr, mf, chroma))

X = [
  [0.829612087, 0.829612087],
  [0.829612087, 0.829612087],
  [0.829612087, 0.829612087],
  [0.829612087, 0.829612087]
]
Y = [
  [1],
  [1],
  [0],
  [0]
]
alpha = .001
print(gradient_descent(X, Y, alpha))