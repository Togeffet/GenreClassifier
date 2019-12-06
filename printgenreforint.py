import os

def genretostring (num):
  indir = "music/train"
  #Get directory list
  dirs = os.listdir(indir)

  for directory in dirs:
    if not os.path.isdir(indir + "/" + directory):
      dirs.remove(directory)


  dirs.sort()

  if num < dirs.count:
    return dirs[int(num)]
  else:
    return "Int " + num + " out of genre bounds"