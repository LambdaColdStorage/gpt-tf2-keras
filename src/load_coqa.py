import hashlib
import os
import random
import re
import json
import ftfy


import tensorflow as tf


class Sampler(object):
    def __init__(self, mode, data_path, enc, n_ctx):
        self.mode = mode
        self.data_path = data_path
        self.enc = enc
        with open(self.data_path + '/coqa-{}-v1.0.json'.format(mode), 'r') as f:
            self.data = json.load(f)["data"]
        self.num_samples = len(self.data)
        self.n_ctx = n_ctx

    def sample(self,):
        for i, x in enumerate(self.data):

            len_sample = 0

            story = x['story']
            questions = x['questions']
            answers = x['answers']

            story = ftfy.fix_text(story)
            story = story.strip()
            enc_story = self.enc.encode(story)

            enc_qa = []
            for q, a in zip(questions, answers):
                enc_q = self.enc.encode(ftfy.fix_text(
                    "\nQ: " + q['input_text']))
                enc_a = self.enc.encode(ftfy.fix_text(
                    "\nA: " + a['input_text']))
                enc_qa += enc_q + enc_a


            if len(enc_story) > self.n_ctx - len(enc_qa):
                enc_story = enc_story[:self.n_ctx - len(enc_qa)]

            enc_story = enc_story + enc_qa
            yield enc_story, enc_story[1:]


def create_dataset(mode, enc, length, dataset_path, batch_size, steps_per_epoch=None, num_epoch=None, output_length=None):
    
    data_sampler = Sampler(mode, dataset_path, enc, length)

    ds = tf.data.Dataset.from_generator(
        data_sampler.sample,
        (tf.int32, tf.int32),
        (tf.TensorShape([None]), tf.TensorShape([None]))
        )

    ds = ds.repeat(num_epoch).shuffle(buffer_size=steps_per_epoch).batch(batch_size, drop_remainder=True)

    return ds