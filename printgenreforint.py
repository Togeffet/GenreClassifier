def genretostring (num):
  if num == 1:
    return 'alternative'
  elif num == 2:
    return 'classical'
  elif num == 3:
    return 'country'
  elif num == 4:
    return 'electronic'
  elif num == 5:
    return 'jazz'
  elif num == 6:
    return 'pop'
  elif num == 7:
    return 'rap'
  elif num == 8:
    return 'rock'
  else:
    return 'No genre exists for param ' + str(num)