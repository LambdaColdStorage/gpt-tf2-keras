import argparse
import json
import importlib

import tensorflow as tf

from src import encoder

parser = argparse.ArgumentParser(description='Input argument parser.')

parser.add_argument('--task', type=str, help='name of task',
                    choices=['texgen', 'qa', 'summary'],
                    default='texgen')

parser.add_argument('--model', type=str, help='name of model',
                    choices=['124M', '355M', '774M', '1558M'],
                    default='124M')

parser.add_argument('--model_ckpt', type=str, help='path of model checkpoint')

parser.add_argument('--json_hparams', type=str, help='path to the json of hyper parameters')

parser.add_argument('--json_encoder', type=str, help='path to the json of encoder')

parser.add_argument('--vocab_bpe', type=str, help='path to the vocabulary bpe')

parser.add_argument('--eager', help='flag to turn on/off eager mode', action='store_true')

parser.add_argument('--nucleus', help='flag to turn on/off nucleus sampling', action='store_true')

parser.add_argument('--top_p', type=float, help='cut off probablity for nucleus sampling',
                    default=1.0)

parser.add_argument('--top_k', type=int, help='cut off ranking for top K sampling',
                    default=2)

parser.add_argument('--temperature', type=float, help='temperature in text generation. Higher temperature creates more randomness in the results.',
                    default=1.0)

parser.add_argument('--dataset_path', type=str, help='path to dataset')

parser.add_argument('--num_epoch', type=int, help='number of training epochs',
                    default=4)

parser.add_argument('--base_lr', type=float, help='base learning rate',
                    default=0.001)

parser.add_argument('--decay_lr', type=float, help='learning rate decay rate',
                    default=0.1)

parser.add_argument('--decay_epoch', type=str, help='epoches to decay learning rate',
                    default='2,3')

parser.add_argument('--steps_per_epoch', type=int, help='number of training step for each epoch',
                    default=100)

parser.add_argument('--batch_size', type=int, help='batch size',
                    default=2)

parser.add_argument('--length', type=int, help='length of input sequence (number of tokens)',
                    default=1024)

parser.add_argument('--data_loader', type=str, help='type of dataset',
                    choices=['text', 'cnndm', 'coqa'])

args = parser.parse_args()


def main():

    if not args.json_hparams:
        print('json_hparams must be provided.')
        print('quit program.')
        exit()

    if not args.json_encoder:
        print('json_encoder must be provided.')
        print('quit program.')
        exit()

    if not args.vocab_bpe:
        print('vocab.bpe must be provided.')
        print('quit program.')
        exit()

    with open(args.json_hparams) as f:
        hparams = json.load(f)

    if not args.eager:
        tf.compat.v1.disable_eager_execution()

    n_vocab = hparams['n_vocab']
    n_ctx = hparams['n_ctx']
    n_embd = hparams['n_embd']
    n_head = hparams['n_head'] 
    n_layer = hparams['n_layer']

    enc = encoder.get_encoder(args.json_encoder, args.vocab_bpe)

    ds = importlib.import_module(
      "src.load_" + args.data_loader).create_dataset(
      enc, args.length, args.dataset_path, args.batch_size)

    # for value in ds.take(2):
    #   print(value)


if __name__ == '__main__':
    main()