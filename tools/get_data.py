import os, urllib.request, gzip, logging

DATA_DIR = 'data'

URL_LIST = [
    ('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', 'train-images-idx3-ubyte.gz'),
    ('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', 'train-labels-idx1-ubyte.gz'),
    ('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', 't10k-images-idx3-ubyte.gz'),
    ('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', 't10k-labels-idx1-ubyte.gz')
]

def maybe_download(url, filepath):
    if os.path.exists(filepath):
        logging.info(f'{filepath} already exists, skipping download')
    else:
        logging.info(f'Downloading {filepath} from {url}. Please wait ...')
        filepath, _ = urllib.request.urlretrieve(url, filepath)
        logging.info('Succesful Download. See : {}'.format(filepath))
        filepath_root, extension = os.path.splitext(filepath)
        if extension == '.gz':
            filepath_extracted = filepath_root
        with gzip.open(filepath, 'rb') as f:
            contents = f.read()
        with open(filepath_extracted, 'wb') as f:
            f.write(contents)

logging.basicConfig(level=logging.INFO)

os.makedirs(DATA_DIR, exist_ok=True)

for (url, filename) in URL_LIST:
    maybe_download(url, os.path.join(DATA_DIR, filename))
