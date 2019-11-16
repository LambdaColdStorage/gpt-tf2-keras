# gpt-tf2
TensorFlow 2 implementation of GTP2, with examples for fine tuning


### Setup

Software 

```
virtualenv -p /usr/bin/python3.6 venv
. venv/bin/activate

pip install -r requirements.txt
```

Hardware

GPT2 is very GPU memory intensive. Here is the minimal requirements for models of different sizes:

* 124M: 11GB
* 355M: 24GB
* 774M: 48GB
* 1558M: not possible on a single GPU.

### Quick Example

Conditional Text Generation


Unconditional Text Generation



# Fine Tuning Examples

### Text Generation

The first application is to fine tune GPT2 to generate text of a particular "style". We used two examples here: the screenplay of `Kill Bill`, and the first five books of `A Song of Ice and Fire`. The training data is stored as a single `txt` file. As a fun test, we will condition text generation by the starter sentence `She picked up the sword` and see if any interesting output can be created by the finetuned model.

First, let see what the pre-trained model (no finetuning) produces:

```
python inference.py \
--model_path=models/355M/model.ckpt \
--json_hparams=models/355M/hparams.json \
--json_encoder=models/355M/encoder.json \
--vocab_bpe=models/355M/vocab.bpe \
--nucleus \
--top_p=0.8 \
--temperature=1.0 \
--output_length=200 \
--starter='She picked up the sword'
```

This is one example output, which is an snippet of fluent English writing, made-up characters and somewhat semi-coherent context.

```
She picked up the sword and went out again, and a second later he entered into the castle.

Yikes.

The moment the shadow disappeared, Qin Yu discovered that there was no sign of the sword before he arrived there.

"Bastard bastard. Don't underestimate me." An old man with a nose wrinkled his brows, "I'll personally help you escape from here!"

That was as expected. A normal and upright person would be strong, but his strength was limited. It wasn't a difficult matter. This didn't have any meaning, just to be able to survive, but had to be said.

Qin Yu couldn't help but take a deep breath. "Bastard!"

Qin Yu held onto the sword as he rushed forward.

He was starting to get worried.

In the end, when Qin Yu left Qin Yu's side, Qin Yu had become ill and would eventually pass away, with no further mention
```

To fine tune GPT2 for text generation, simply specify the model (size, pre-trained ckpt, json files for model hyper parameters and the encoder, the byte-pair-encoding of the vocabular) and the training data (path to the text file and the type of loader).

The following command fine tunes the 355M model on `Kill Bill` for 4 epoch, where each epoch has 500 pieces (1024 tokens each) of text randomly sampled from the screenplay. 

```
python finetune.py \
--model=355M \
--model_ckpt=models/355M/model.ckpt \
--json_hparams=models/355M/hparams.json \
--json_encoder=models/355M/encoder.json \
--vocab_bpe=models/355M/vocab.bpe \
--output_name=killbill_355M.h5 \
--dataset_path=dataset/killbill.txt \
--data_loader=text \
--num_epoch=4 \
--steps_per_epoch=500
```

To test the finetuned model, we generate 200 tokens using `nucleus sampling` with `top_p=1.0` and `temperature=1.0`:

```
python inference.py \
--model_path=output/killbill_355M.h5 \
--json_hparams=models/355M/hparams.json \
--json_encoder=models/355M/encoder.json \
--vocab_bpe=models/355M/vocab.bpe \
--nucleus \
--top_p=1.0 \
--temperature=1.0 \
--output_length=200 \
--starter='She picked up the sword'
```

This is the result:

```
She picked up the sword, and is slowly dragging it over her head.

TILL...

They lock eyes...

...Yuki yells to her in Japanese;



                    YUKI (JAPANESE)
          ...You can kill yourself.

Yuki giggles.



                      YUKI (JAPANESE)
          Just don't make me kill you.



                    THE BRIDE (JAPANESE)
          Okay, I want to see how    good you really are.
```

As you can see, the output start looks a lot more like a screenplay, with the correct format and characters from `Kill Bill` (the bride and YUKI). 

Let's now finetune GPT2 on R.R. Martin's five books of `A Song of Ice and Fire`:

```
python finetune.py \
--model=355M \
--model_ckpt=models/355M/model.ckpt \
--json_hparams=models/355M/hparams.json \
--json_encoder=models/355M/encoder.json \
--vocab_bpe=models/355M/vocab.bpe \
--output_name=CompleteRRMartin_355M.h5 \
--dataset_path=dataset/CompleteRRMartin.txt \
--data_loader=text \
--num_epoch=4 \
--steps_per_epoch=500

python inference.py \
--model_path=output/CompleteRRMartin_355M.h5 \
--json_hparams=models/355M/hparams.json \
--json_encoder=models/355M/encoder.json \
--vocab_bpe=models/355M/vocab.bpe \
--nucleus \
--top_p=1.0 \
--temperature=1.0 \
--output_length=200 \
--starter='She picked up the sword'
```

And this is one example output:

```
She picked up the sword before her she could drop it. “When I send a message at Horos, 
I tell him that Horos was to take the black wolf, save my hands. A soul goes in to watch over 
my arms, arms, and feet too. Beneath the sense of fear, every man is fearful, undead. 
The Wall gives fathers away. Only the echoing voice allows the dead to emerge from our shadows, 
unseen. The leaves will send them away, and thrice that man who stirs the dead will alter and grow stronger.”

She had crossed the narrow sea before, exporting raw from the coast that had been her home. 
Leaving Horos only had proved to be offering no sort of comfort to Tyrion. Like the despairs, 
most of what he said ended according to whim, which made no sense to most of Whores’s ship. 
And her proud voice, Yezzan’s, woken only when creak made of bells.
```

Amazingly, the output has many elements from `A Song of Ice and Fire`. It talks about the `wall`, `black wolf`, `the dead`, `the nerrow sea`. It also mentioned the popular character `Tyrion`, and a less significant character `Yezzan` (an extremely wealthy slave trader, and one of the Wise Masters from Yunkai). It also invented a new character/place named `Horos`. 



### Text Summarization

```
python finetune.py \
--model=355M \
--model_ckpt=models/355M/model.ckpt \
--json_hparams=models/355M/hparams.json \
--json_encoder=models/355M/encoder.json \
--vocab_bpe=models/355M/vocab.bpe \
--output_name=cnndm_355M.h5 \
--dataset_path=/home/ubuntu/data/summarization \
--data_loader=cnndm \
--num_epoch=4 \
--steps_per_epoch=100


python inference.py \
--model_path=output/cnndm_355M.h5 \
--json_hparams=models/355M/hparams.json \
--json_encoder=models/355M/encoder.json \
--vocab_bpe=models/355M/vocab.bpe \
--nucleus \
--top_p=1.0 \
--temperature=1.0 \
--output_length=200 \
--starter="In the Second Age of Middle-earth, the lords of Elves, Dwarves, and Men are given Rings of Power. Unbeknownst to them, the Dark Lord Sauron forges the One Ring in Mount Doom, infusing into it a great part of his power to dominate, through it and at a distance, the other Rings, so he might conquer Middle-earth. A final alliance of men and elves battles Sauron\'s forces in Mordor, where Prince Isildur of Gondor severs Sauron\'s finger, and the Ring with it, thereby destroying his physical form. With Sauron\'s first defeat, the Third Age of Middle-earth begins. Unfortunately, the Ring\'s influence corrupts Isildur, and, rather than destroy the Ring, Isildur takes it for himself. Isildur is later killed by Orcs, and the Ring is lost for 2,500 years, until it is found by Gollum, who owns it for five centuries. The Ring is then found by a hobbit named Bilbo Baggins, who turns invisible when he puts it on, but is unaware of its history. \n@highlight\n"


python inference.py \
--model_path=models/355M/model.ckpt \
--json_hparams=models/355M/hparams.json \
--json_encoder=models/355M/encoder.json \
--vocab_bpe=models/355M/vocab.bpe \
--nucleus \
--top_p=1.0 \
--temperature=1.0 \
--output_length=200 \
--starter="In the Second Age of Middle-earth, the lords of Elves, Dwarves, and Men are given Rings of Power. Unbeknownst to them, the Dark Lord Sauron forges the One Ring in Mount Doom, infusing into it a great part of his power to dominate, through it and at a distance, the other Rings, so he might conquer Middle-earth. A final alliance of men and elves battles Sauron\'s forces in Mordor, where Prince Isildur of Gondor severs Sauron\'s finger, and the Ring with it, thereby destroying his physical form. With Sauron\'s first defeat, the Third Age of Middle-earth begins. Unfortunately, the Ring\'s influence corrupts Isildur, and, rather than destroy the Ring, Isildur takes it for himself. Isildur is later killed by Orcs, and the Ring is lost for 2,500 years, until it is found by Gollum, who owns it for five centuries. The Ring is then found by a hobbit named Bilbo Baggins, who turns invisible when he puts it on, but is unaware of its history. \n@highlight\n"

```


### Reading Comprehension


```
python finetune.py \
--model=355M \
--model_ckpt=models/355M/model.ckpt \
--json_hparams=models/355M/hparams.json \
--json_encoder=models/355M/encoder.json \
--vocab_bpe=models/355M/vocab.bpe \
--output_name=coqa_355M.h5 \
--dataset_path=/home/ubuntu/data/coqa \
--data_loader=coqa \
--num_epoch=4 \
--steps_per_epoch=100


python inference.py \
--model_path=output/coqa_355M.h5 \
--json_hparams=models/355M/hparams.json \
--json_encoder=models/355M/encoder.json \
--vocab_bpe=models/355M/vocab.bpe \
--nucleus \
--top_p=1.0 \
--temperature=1.0 \
--output_length=200 \
--starter="The 2008 Summer Olympics torch relay was run from March 24 until August 8, 2008, prior to the 2008 Summer Olympics, with the theme of \“one world, one dream\”. Plans for the relay were announced on April 26, 2007, in Beijing, China. The relay, also called by the organizers as the “Journey of Harmony”, lasted 129 days and carried the torch 137,000 km (85,000 mi) – the longest distance of any Olympic torch relay since the tradition was started ahead of the 1936 Summer Olympics. After being lit at the birthplace of the Olympic Games in Olympia, Greece on March 24, the torch traveled to the Panathinaiko Stadium in Athens, and then to Beijing, arriving on March 31. From Beijing, the torch was following a route passing through six continents. The torch has visited cities along the Silk Road, symbolizing ancient links between China and the rest of the world. The relay also included an ascent with the flame to the top of Mount Everest on the border of Nepal and Tibet, China from the Chinese side, which was closed specially for the event. \n Q: What was the theme? \n A: “one world, one dream”. \n Q: What was the length of the race? \n A: 137,000 km \n Q: Was it larger than previous ones? \n A: No \n Q: Where did the race begin? \n A: Olympia, Greece \n Q: Is there anything notable about that place? \n A: birthplace of Olympic Games \n Q: Where did they go after? \n A: Athens \n Q: How many days was the race? \n A: seven \n Q: Did they visit any notable landmarks? \n A: Panathinaiko Stadium \n Q: And did they climb any mountains? \n A:"

```

