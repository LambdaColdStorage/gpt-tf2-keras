import hashlib
import os
import random
import re
import ftfy

import tensorflow as tf
# tf.compat.v1.disable_eager_execution()

def clean_up_start(text):
    if text[:2] == 'By':
        text = '\n'.join(text.split('\n')[2:])
    text = re.split(r'\(CNN\) +--', text)[-1]
    text = re.split(r"\(CNN\)", text[:100])[-1]+text[100:]
    text = re.sub(r"^and \w+\n", "", text)
    text = re.split(r".*UPDATED:\s+[0-9]{2}:[0-9]{2}.*[2011|2012|2013|2014|2015]", text)[-1]
    text = text.replace('’', "'")
    text = text.replace('‘', "'")
    return text.strip()


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s)
    return h.hexdigest()


def get_path_of_url(url):
    if 'dailymail.co.uk' in url or 'mailonsunday.ie' in url or 'lib.store.yahoo.net' in url:
        site = 'dailymail'
    else:
        assert 'cnn.com' in url or 'cnn.hk' in url, url
        site = 'cnn'
    url_hash = hashhex(url.encode('utf-8'))
    return '{0}/stories/{1}.story'.format(site, url_hash)


def cnndm_generator(mode='valid', enc=None, seed=0, shuffle=False, comm=None):
    # data originally from https://github.com/abisee/cnn-dailymail
    if mode == 'valid':
        mode = 'val'

    with open('/home/ubuntu/data/summarization/url_lists/all_{}.txt'.format(mode), 'r') as f:
        urls = [line.strip() for line in f]

    urls_dir = '/home/ubuntu/data/summarization'

    if shuffle:
        random.seed(seed)
        random.shuffle(urls)

    for i, url in enumerate(urls):
        path = os.path.join(urls_dir, get_path_of_url(url))
        text = open(path).read()
        text = clean_up_start(text)
        text = ftfy.fix_text(text)

        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.split('@highlight')[0].strip()

        yield text


class Sampler(object):
    def __init__(self, mode, data_path, enc, n_ctx, output_length):
        self.mode = mode
        self.data_path = data_path
        self.enc = enc
        with open(self.data_path + '/url_lists/all_{}.txt'.format(mode), 'r') as f:
            self.urls = [line.strip() for line in f]
        self.num_samples = len(self.urls)
        self.n_ctx = n_ctx
        self.output_length = output_length

    def sample(self,):

        for i, url in enumerate(self.urls):
            path = os.path.join(self.data_path, get_path_of_url(url))
            text = open(path).read()
            text = clean_up_start(text)
            text = ftfy.fix_text(text)

            text = re.sub(r"\n{3,}", "\n\n", text).split('@highlight')
            raw_text = text[0].strip()
            enc_text = self.enc.encode(raw_text)
            enc_seperator = self.enc.encode('\nTL;DR:')

            if self.mode == 'train':
                idx_highlight = random.randint(1, len(text) - 1)
                enc_highlight = self.enc.encode(text[idx_highlight].strip())

                # '<|endoftext|>'
                enc_end = [50256]

                enc_output = enc_seperator + enc_highlight + enc_end

                if len(enc_text) > self.n_ctx - len(enc_output):
                    enc_text = enc_text[:self.n_ctx - len(enc_output)]

                enc_input = enc_text + enc_output
                yield enc_input, enc_input[1:]

            else:                
                if len(enc_text) > self.n_ctx - self.output_length - len(enc_seperator):
                    enc_text = enc_text[:self.n_ctx - self.output_length - len(enc_seperator)]

                enc_input = enc_text + self.enc.encode('\nTL;DR:')

                yield enc_input, self.num_samples



def create_dataset(mode, enc, length, dataset_path, batch_size, steps_per_epoch=None, num_epoch=None, output_length=None):
    
    data_sampler = Sampler(mode, dataset_path, enc, length, output_length)

    if mode == 'train':
        ds = tf.data.Dataset.from_generator(
            data_sampler.sample,
            (tf.int32, tf.int32),
            (tf.TensorShape([None]), tf.TensorShape([None]))
            )
        ds = ds.repeat(num_epoch).shuffle(buffer_size=steps_per_epoch).batch(batch_size, drop_remainder=True)
    else:
        ds = data_sampler.sample()
    return ds