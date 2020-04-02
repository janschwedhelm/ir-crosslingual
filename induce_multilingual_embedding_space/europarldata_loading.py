import wget
import gzip

file_url = 'http://opus.nlpl.eu/download.php?f=Europarl/v8/tmx/de-en.tmx.gz'
wget.download(file_url, 'de-en.tmx.gz')

import shutil
with gzip.open('de-en.tmx.gz', 'rb') as f_in, open('de-en.txt', 'wb') as f_out:
  shutil.copyfileobj(f_in, f_out)
