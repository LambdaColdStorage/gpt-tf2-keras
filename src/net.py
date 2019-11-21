import json

import tensorflow as tf
from tensorflow import keras
from src.layers import EmbeddingSim, EmbeddingRet, PositionEmbedding, LayerNormalization, _get_encoder_component, gelu

# tf.compat.v1.disable_eager_execution()


def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(
    labels, logits[:, :-1, :], from_logits=True)

def create_model(args):

    if not args.json_hparams:
        print('json_hparams must be provided.')
        print('quit program.')
        exit()

    with open(args.json_hparams) as f:
        hparams = json.load(f)

    n_vocab = hparams['n_vocab']
    n_ctx = hparams['n_ctx']
    n_embd = hparams['n_embd']
    n_head = hparams['n_head'] 
    n_layer = hparams['n_layer']
    
    batch_size = args.batch_size

    input_layer = keras.layers.Input(
        batch_shape=(batch_size, None),
        name='Input',
    )

    embed_token, embeddings = EmbeddingRet(
        input_dim=n_vocab,
        output_dim=n_embd,
        mask_zero=False,
        name='Embed-Token',
    )(input_layer)

    embed_token_pos = PositionEmbedding(
        input_dim=n_ctx,
        output_dim=n_embd,
        mode='add',
        name='Embed-Token-Pos',
    )(embed_token)

    last_layer = embed_token_pos
    for i in range(n_layer):
        last_layer = _get_encoder_component(
            name='Encode-%d' % i,
            input_layer=last_layer,
            head_num=n_head,
            hidden_dim=n_embd * 4,
            attention_activation=None,
            feed_forward_activation=gelu,
        )

    norm_layer = LayerNormalization(
        name='Norm',
    )(last_layer)

    output_layer = EmbeddingSim(
        use_bias=False,
        name='Output',
    )([norm_layer, embeddings])


    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    return model


def load_weights(model, args):

    if not args.json_hparams:
        print('json_hparams must be provided.')
        print('quit program.')
        exit()

    with open(args.json_hparams) as f:
        hparams = json.load(f)

    n_vocab = hparams['n_vocab']
    n_ctx = hparams['n_ctx']
    n_embd = hparams['n_embd']
    n_head = hparams['n_head'] 
    n_layer = hparams['n_layer']

    layer = model.get_layer(name='Embed-Token')
    layer.set_weights([
        tf.train.load_variable(args.model_ckpt, 'model/wte:0'),
    ])
    layer.trainable=False

    layer = model.get_layer(name='Embed-Token-Pos')
    layer.set_weights([
        tf.train.load_variable(args.model_ckpt, 'model/wpe:0')[:, :],
    ])
    layer.trainable=False

    for i in range(n_layer):
        model.get_layer(name='Encode-%d-MultiHeadAtt-Norm' % i).set_weights([
            tf.train.load_variable(args.model_ckpt, 'model/h%d/ln_1/g:0' % i),
            tf.train.load_variable(args.model_ckpt, 'model/h%d/ln_1/b:0' % i),
        ])
        kernel = tf.train.load_variable(args.model_ckpt, 'model/h%d/attn/c_attn/w:0' % i)[0]
        bias = tf.train.load_variable(args.model_ckpt, 'model/h%d/attn/c_attn/b:0' % i)
        model.get_layer(name='Encode-%d-MultiHeadAtt' % i).set_weights([
            kernel[:, :n_embd],
            bias[:n_embd],
            kernel[:, n_embd:-n_embd],
            bias[n_embd:-n_embd],
            kernel[:, -n_embd:],
            bias[-n_embd:],
            tf.train.load_variable(args.model_ckpt, 'model/h%d/attn/c_proj/w:0' % i)[0],
            tf.train.load_variable(args.model_ckpt, 'model/h%d/attn/c_proj/b:0' % i),
        ])
        model.get_layer(name='Encode-%d-FeedForward-Norm' % i).set_weights([
            tf.train.load_variable(args.model_ckpt, 'model/h%d/ln_2/g:0' % i),
            tf.train.load_variable(args.model_ckpt, 'model/h%d/ln_2/b:0' % i),
        ])
        model.get_layer(name='Encode-%d-FeedForward' % i).set_weights([
            tf.train.load_variable(args.model_ckpt, 'model/h%d/mlp/c_fc/w:0' % i)[0],
            tf.train.load_variable(args.model_ckpt, 'model/h%d/mlp/c_fc/b:0' % i),
            tf.train.load_variable(args.model_ckpt, 'model/h%d/mlp/c_proj/w:0' % i)[0],
            tf.train.load_variable(args.model_ckpt, 'model/h%d/mlp/c_proj/b:0' % i),
        ])
    layer = model.get_layer(name='Norm')
    layer.set_weights([
        tf.train.load_variable(args.model_ckpt, 'model/ln_f/g:0'),
        tf.train.load_variable(args.model_ckpt, 'model/ln_f/b:0'),
    ])
    layer.trainable=False

    return model


def create_schedule(args):
    decay_epochs = [int(x) for x in args.decay_epochs.split(',')]

    def schedule(epoch):
        learning_rate = args.base_lr * args.batch_size
        for e in decay_epochs:

            if epoch >= e:
                learning_rate = args.decay_lr * learning_rate
            else:
                break

        return learning_rate

    return schedule