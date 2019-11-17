import os
import argparse
import importlib


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler


from src import encoder
from src import net


parser = argparse.ArgumentParser(description='Input argument parser.')

parser.add_argument('--model', type=str, help='name of model',
                    choices=['124M', '355M', '774M', '1558M'],
                    default='124M')

parser.add_argument('--model_ckpt', type=str, help='path of model checkpoint')

parser.add_argument('--json_hparams', type=str, help='path to the json of hyper parameters')

parser.add_argument('--json_encoder', type=str, help='path to the json of encoder')

parser.add_argument('--vocab_bpe', type=str, help='path to the vocabulary bpe')

parser.add_argument('--eager', help='flag to turn on/off eager mode', action='store_true')

parser.add_argument('--dataset_path', type=str, help='path to dataset')

parser.add_argument('--num_epoch', type=int, help='number of training epochs',
                    default=4)

parser.add_argument('--base_lr', type=float, help='base learning rate',
                    default=0.001)

parser.add_argument('--decay_lr', type=float, help='learning rate decay rate',
                    default=0.1)

parser.add_argument('--decay_epochs', type=str, help='epoches to decay learning rate',
                    default='1000,10000')

parser.add_argument('--steps_per_epoch', type=int, help='number of training step for each epoch',
                    default=100)

parser.add_argument('--batch_size', type=int, help='batch size',
                    default=1)

parser.add_argument('--length', type=int, help='length of input sequence (number of tokens)',
                    default=1024)

parser.add_argument('--data_loader', type=str, help='type of dataset',
                    choices=['text', 'cnndm', 'coqa'])

parser.add_argument('--output_name', type=str, help='name of output model')

args = parser.parse_args()


def main():

    if not args.eager:
        tf.compat.v1.disable_eager_execution()

    if not args.json_encoder:
        print('json_encoder must be provided.')
        print('quit program.')
        exit()

    if not args.vocab_bpe:
        print('vocab.bpe must be provided.')
        print('quit program.')
        exit()

    if not os.path.exists('output'):
        os.makedirs('output')

    enc = encoder.get_encoder(args.json_encoder, args.vocab_bpe)

    ds = importlib.import_module(
        "src.load_" + args.data_loader).create_dataset(
        enc, args.length, args.dataset_path, args.batch_size, args.steps_per_epoch, args.num_epoch)

    # for value in ds.take(10):
    #     x = enc.decode(value[0][0].numpy())
    #     print(x)
    #     print(len(value[0][0]))
    #     input("Press Enter to continue...")

    # exit()

    model = net.create_model(args)

    # restore weight
    model = net.load_weights(model, args)

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=net.loss
    )

    # fine tune
    model.fit(ds,
              epochs=args.num_epoch,
              steps_per_epoch=args.steps_per_epoch,
              callbacks=[LearningRateScheduler(net.create_schedule(args))])

    model.save(os.path.join('output', args.output_name))

if __name__ == '__main__':
    main()