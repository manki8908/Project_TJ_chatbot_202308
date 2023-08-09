def timeSince(since):

    import math
    import time

    now = time.time()
    s = now - since
    m = math.floor(s/60)
    s -= m*60
   
    return '%dm %ds' % (m, s)
