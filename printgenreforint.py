def genretostring (num):
  if num == 0:
    return 'alternative'
  elif num == 1:
    return 'classical'
  elif num == 2:
    return 'country'
  elif num == 3:
    return 'electronic'
  elif num == 4:
    return 'jazz'
  elif num == 5:
    return 'pop'
  elif num == 6:
    return 'rap'
  elif num == 7:
    return 'rock'
  else:
    return 'No genre exists for param ' + str(num)