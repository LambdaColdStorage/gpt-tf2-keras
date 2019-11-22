import numpy as np
import argparse


import tensorflow as tf
from tensorflow import keras


from src.layers import EmbeddingSim, EmbeddingRet, PositionEmbedding, LayerNormalization, _get_encoder_component, gelu, ScaledDotProductAttention, MultiHeadAttention, FeedForward
from src import encoder
from src import net
from src import utils

tf.compat.v1.disable_eager_execution()

parser = argparse.ArgumentParser(description='Input argument parser.')

parser.add_argument('--model_path', type=str, help='path of model')

parser.add_argument('--json_hparams', type=str, help='path to the json of hyper parameters')

parser.add_argument('--json_encoder', type=str, help='path to the json of encoder')

parser.add_argument('--vocab_bpe', type=str, help='path to the vocabulary bpe')

parser.add_argument('--nucleus', help='flag to turn on/off nucleus sampling', action='store_true')

parser.add_argument('--top_p', type=float, help='cut off probablity for nucleus sampling',
                    default=1.0)

parser.add_argument('--top_k', type=int, help='cut off ranking for top K sampling',
                    default=2)

parser.add_argument('--temperature', type=float, help='temperature in text generation. Higher temperature creates more randomness in the results.',
                    default=1.0)

parser.add_argument('--batch_size', type=int, help='batch size',
                    default=1)

parser.add_argument('--output_length', type=int, help='length of output sequence (number of tokens)',
                    default=100)

parser.add_argument('--starter', type=str, help='starter sentence')

args = parser.parse_args()


# @tf.function
def sample(model, input_data, start_length, flag_stop,
           output_length, batch_size, nucleus, top_p, top_k,
           temperature):

    for shift in range(output_length):
        output_data = model.predict(np.array(input_data))

        for index in range(batch_size):

            if not flag_stop[index]:
                probs = [(prob, i) for i, prob in enumerate(output_data[index, start_length[index] + shift - 1])]
                probs.sort(reverse=True)
                
                if nucleus:
                    next_token = utils.find_top_p(probs, top_p, temperature)
                else:
                    next_token = utils.find_top_k(probs, top_k, temperature)

                input_data[index].append(next_token)

                if next_token == 50256:
                    flag_stop[index] = True
            else:
                input_data[index].append(50256)
    return input_data


def main():

    if not args.json_encoder:
        print('json_encoder must be provided.')
        print('quit program.')
        exit()

    if not args.vocab_bpe:
        print('vocab.bpe must be provided.')
        print('quit program.')
        exit()

    args.starter = args.starter.replace("\\n", "\n")
    args.starter = args.starter.replace("\\'", "'")

    # print(args.starter)
    # exit()
    
    enc = encoder.get_encoder(args.json_encoder, args.vocab_bpe)

    # load model
    if args.model_path.split('.')[-1] == 'h5':
        model = keras.models.load_model(
            args.model_path,
            custom_objects={'EmbeddingSim': EmbeddingSim,
                            'EmbeddingRet': EmbeddingRet,
                            'PositionEmbedding': PositionEmbedding,
                            'LayerNormalization': LayerNormalization,
                            'ScaledDotProductAttention': ScaledDotProductAttention,
                            'MultiHeadAttention': MultiHeadAttention,
                            'FeedForward': FeedForward,
                            'gelu': gelu,
                            'loss': net.loss})
    elif args.model_path.split('.')[-1] == 'ckpt':
        args.model_ckpt = args.model_path
        model = net.create_model(args)
        model = net.load_weights(model, args)
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=net.loss
        )
    else:
        print('Unrecognized model format: ' + args.model_path.split('.')[-1])
        exit()

    model.trainable = False    

    # prepare input data
    input_data = [enc.encode(args.starter) for i in range(args.batch_size)]
    start_length = [len(data) for data in input_data]
    flag_stop = [False for i in range(args.batch_size)]

    # run inference

    print('Start generation: ... ')
    import time
    start = time.time()

    output = sample(model, input_data, start_length, flag_stop,
                    args.output_length, args.batch_size, args.nucleus,
                    args.top_p, args.top_k, args.temperature)

    end = time.time()
    print(end - start)
    print('------------------------')
    # print result
    for index in range(args.batch_size):
        output = enc.decode(output[index])
        print(output)
        print('--------------------------------------------------')

if __name__ == '__main__':
    main()