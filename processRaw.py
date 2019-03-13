import time
import gzip
import simplejson

def parse(filename):
  f = gzip.open(filename, 'r')
  entry = {}
  for l in f:
    l = l.strip()
    colonPos = l.find(':')
    if colonPos == -1:
      yield entry
      entry = {}
      continue
    eName = l[:colonPos]
    rest = l[colonPos+2:]
    entry[eName] = rest
  yield entry

for e in parse("Beeradvocate.txt.gz"):
  try:
    e['review/appearance'] = float(e['review/appearance'])
    e['review/taste'] = float(e['review/taste'])
    e['review/overall'] = float(e['review/overall'])
    e['review/palate'] = float(e['review/palate'])
    e['review/aroma'] = float(e['review/aroma'])
    e['review/timeUnix'] = int(e['review/time'])
    e.pop('review/time', None)
    try:
      e['beer/ABV'] = float(e['beer/ABV'])
    except Exception as q:
      e.pop('beer/ABV', None)
    e['user/profileName'] = e['review/profileName']
    e.pop('review/profileName', None)
    timeStruct = time.gmtime(e['review/timeUnix'])
    e['review/timeStruct'] = dict(zip(["year", "mon", "mday", "hour", "min", "sec", "wday", "yday", "isdst"], list(timeStruct)))
    import ipdb; ipdb.set_trace()
    print e
  except Exception as q:
    pass
