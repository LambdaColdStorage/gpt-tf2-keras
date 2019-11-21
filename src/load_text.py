import glob
import numpy as np
import os
import random
import tqdm
import csv

import tensorflow as tf
# tf.compat.v1.disable_eager_execution()


def load_dataset(enc, path, combine):
    paths = []
    if os.path.isfile(path):
        # Simple file
        paths.append(path)
    elif os.path.isdir(path):
        # Directory
        for (dirpath, _, fnames) in os.walk(path):
            for fname in fnames:
                paths.append(os.path.join(dirpath, fname))
    else:
        # Assume glob
        paths = glob.glob(path)

    token_chunks = []
    raw_text = ''
    for path in tqdm.tqdm(paths):
        if path.endswith('.npz'):
            # Pre-encoded
            with np.load(path) as npz:
                for item in npz.files:
                    token_chunks.append(npz[item])
        elif path.endswith('.csv'):
            start_token = "<|startoftext|>"
            end_token = "<|endoftext|>"
            with open(path, 'r', encoding='utf8', errors='ignore') as fp:
                fp.readline()   # skip header
                reader = csv.reader(fp)
                for row in reader:
                    raw_text += start_token + row[0] + end_token + "\n"
        else:
            # Plain text
            with open(path, 'r', encoding='utf8', errors='ignore') as fp:
                raw_text += fp.read()
            if len(raw_text) >= combine:
                tokens = np.stack(enc.encode(raw_text))
                token_chunks.append(tokens)
                raw_text = ''
            else:
                raw_text += '<|endoftext|>'
    if raw_text:
        tokens = np.stack(enc.encode(raw_text))
        token_chunks.append(tokens)
    return token_chunks


def binary_search(f, lo, hi):
    if f(lo) or not f(hi):
        return None
    while hi > lo + 1:
        mid = (lo + hi) // 2
        if f(mid):
            hi = mid
        else:
            lo = mid
    return hi


class Sampler(object):
    """Fairly samples a slice from a set of variable sized chunks.
    'Fairly' means that the distribution is the same as sampling from one concatenated chunk,
    but without crossing chunk boundaries."""

    def __init__(self, chunks, length):
        self.chunks = chunks
        self.total_size = sum(chunk.shape[0] for chunk in chunks)
        self.boundaries = [0]
        self.length = length
        for i in range(len(chunks)):
            self.boundaries.append(self.boundaries[-1] + chunks[i].shape[0])

    def sample(self,):
        assert self.length < self.total_size // len(
            self.chunks
        ), "Dataset files are too small to sample {} tokens at a time".format(
            self.length)
        while True:
            index = random.randint(0, self.total_size - self.length - 1)
            i = binary_search(lambda j: self.boundaries[j] > index, 0,
                              len(self.boundaries) - 1) - 1
            if self.boundaries[i + 1] > index + self.length:
                within_chunk = index - self.boundaries[i]
                output = self.chunks[i][within_chunk:within_chunk + self.length]
                # The ground truth is the second to the last elements of the input
                yield output, output[1:]


def create_dataset(mode, enc, length, dataset_path, batch_size, steps_per_epoch=None, num_epoch=None, output_length=None):
    
    combine = 50000

    chunks = load_dataset(enc, dataset_path, combine)

    data_sampler = Sampler(chunks, length)

    ds = tf.data.Dataset.from_generator(
        data_sampler.sample,
        (tf.int32, tf.int32),
        (tf.TensorShape([length]), tf.TensorShape([length - 1]))
        )

    ds = ds.repeat(num_epoch).batch(batch_size, drop_remainder=True)

    return ds