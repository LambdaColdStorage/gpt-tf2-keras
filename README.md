# gpt-tf2
TensorFlow 2 implementation of GTP2 for fine-tuning on a single GPU.

1. [Setup](#setup)
	1. [Software](#software)
	2. [Hardware](#hardware)
	3. [Pre-trained Models](#pre-trained)
	4. [Data](#data)
2. [Acknowledgement](#acknowledgement)
3. [Examples](#examples)
	1. [Text Generation](#text-generation)
	2. [Text Summarization](#text-summarization)
	3. [Conversational Question and Answer](#conversational-qa)
4. [Small Models](#small-models)
	1. [Text Generation 124M](#text-generation-124M)
	2. [Text Summarization 124M](#text-summarization-124M)
	3. [Conversational Question and Answer 124M](#conversational-qa-124M)			
5. [Medium Models](#medium-models)
	1. [Text Generation 355M](#text-generation-355M)
	2. [Text Summarization 355M](#text-summarization-355M)
	3. [Conversational Question and Answer 355M](#conversational-qa-355M)		
6. [Large Models](#large-models)
	1. [Text Generation 774M](#text-generation-774M)
	2. [Text Summarization 774M](#text-summarization-774M)
	3. [Conversational Question and Answer 774M](#conversational-qa-774M)	
7. [Evaluation](#evaluation)
	1. [Text Summarization (CNNDM)](#evaluation-cnndm)

## Setup <a name="setup"></a>

### Software <a name="software"></a>

```
virtualenv -p /usr/bin/python3.6 venv
. venv/bin/activate

pip install -r requirements.txt
```

### Hardware <a name="hardware"></a>

GPT2 is very GPU memory intensive. Here is the minimal requirements for models of different sizes:

Training
* 124M: 11GB (1080Ti, 2080Ti etc)
* 355M: 24GB (RTX Titan, RTX Quadro 6000, Tesla V100 etc)
* 774M: 48GB (RTX Quadro 8000)
* 1558M: seems not possible on a single GPU.

Inference
* 124M: 11GB (1080Ti, 2080Ti etc)
* 355M: 11GB (1080Ti, 2080Ti etc)
* 774M: 24GB (RTX Titan, RTX Quadro 6000, Tesla V100 etc)


### Pre-trained Models <a name="pre-trained"></a>

```
python download_model.py model_size (choose from 124M, 355M, 774M and 1558M)
```

### Data <a name="data"></a>

__Text Generation__

The '.txt' are available in the "dataset" folder.

__Text Summarization__

```
sudo apt install default-jre            
sudo apt install openjdk-11-jre-headless
sudo apt install openjdk-8-jre-headless

# Download stories and unzip to /home/ubuntu/data/cnn_stories and /home/ubuntu/data/dailymail_stories
https://cs.nyu.edu/~kcho/DMQA/

# Download Stanford CoreNLP and unzip to home/ubuntu/data/stanford-corenlp-full-2018-10-05/
https://stanfordnlp.github.io/CoreNLP/


git clone https://github.com/abisee/cnn-dailymail

cd cnn-dailymail

export CLASSPATH=/home/ubuntu/data/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar

python make_datafiles.py /home/ubuntu/data/cnn_stories/cnn/stories /home/ubuntu/data/dailymail_stories/dailymail/stories

# copy folders url_lists, cnn_stories/cnn, dailymail_stories/dailymail to /home/ubuntu/data/summarization
```

__Conversational Question and Anwser__


```
# Download training json
https://stanfordnlp.github.io/coqa/

# copy to ~/data/coqa
```

## Acknowledgement <a name="acknowledgement"></a>

This project would not be possible without the guidance and inspiration from these repositories:

[OpenAI GPT2](https://github.com/openai/gpt-2): For pre-trained GPT2 models and examples of running inference with them.

[OpenAI ln-human-preferences](https://github.com/openai/lm-human-preferences): For example of data loader for the `cnn-dailymail` dataset.

[minimaxir](https://github.com/minimaxir/gpt-2-simple): For examples of fine-tuning GPT2 models in TensorFlow 1.14.

[CyberZHG](https://github.com/CyberZHG/keras-gpt-2): For examples of Keras implementation of GPT2 graph and restoring weights from checkpoints.

[OpenNMT](http://opennmt.net/OpenNMT-py): For instructions of evaluating CNNDM results using Rouge Score.

Notice: This repo __does not__ implement the RL based fine-tuning algorithm as described in [this blog](https://openai.com/blog/fine-tuning-gpt-2/). In contrast, we fine-tune the transformer layers using additional datasets for each new application.


## Examples <a name="examples"></a>

### Text Generation <a name="text-generation"></a>

The first application is to fine-tune GPT2 to generate text of a particular "style." We have two examples here: the screenplay of `Kill Bill,` and the first five books of `A Song of Ice and Fire`. Each training dataset is stored as a single `.txt` file. 

For testing, we condition the text generation by the starter sentence `She picked up the sword` and see if the fine-tuned model can create any exciting output.

First, let see what the pre-trained model (no fine-tuning) produces:

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

The paragraph below is an example output: the English is largely fluent, however it presents made-up characters and somewhat semi-coherent story.

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

To fine-tune GPT2 for text generation, we specify the model (size, pre-trained ckpt, json files for model hyperparameters and the encoder, the byte-pair-encoding of the vocabulary) and the training data (path to the text file and the type of loader).

The following command fine-tunes the 355M model on `Kill Bill` for one epoch that has 2000 pieces (1024 tokens each) of text randomly sampled from the screenplay. 

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
--base_lr=0.0001 \
--num_epoch=1 \
--steps_per_epoch=2000
```

To test the fine-tuned model, we generate 200 tokens using `nucleus sampling` with `top_p=1.0` and `temperature=1.0`:

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

As you can see, the output start looks a lot more like a screenplay, with the correct format and characters from `Kill Bill` (THE BRIDE and YUKI). 

Let's now fine-tune GPT2 on R.R. Martin's five books of `A Song of Ice and Fire`:

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
--base_lr=0.0001 \
--num_epoch=1 \
--steps_per_epoch=2000

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

Amazingly, the output uses many concepts from `A Song of Ice and Fire`. It talks about the `wall,` `the black wolf,` `the dead,` and `the nerrow sea.` It also mentioned the popular figure `Tyrion,` and a less significant character `Yezzan` (an extremely wealthy slave trader, and one of the Wise Masters from Yunkai). It also invented a new character/place named `Horos.` 



### Text Summarization <a name="text-summarization"></a>


Our next experiment fine-tunes the GPT2 model for text summarization. The original OpenAI GPT2 blog demonstrated that a pre-trained model (without fine-tuning) has a certain ability to [summarize](https://openai.com/blog/better-language-models/#task5), indicating the training dataset contains some examples where the model learned the concept of "summarization" and the certain keywords to triggle the summarization process. 

As described in the [original paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)(Section 3.6), one only need to add `TL;DR:` to the end of the starter text
for "zero-shot" text summarization:

```
python inference.py \
--model_path=models/355M/model.ckpt \
--json_hparams=models/355M/hparams.json \
--json_encoder=models/355M/encoder.json \
--vocab_bpe=models/355M/vocab.bpe \
--nucleus \
--top_p=1.0 \
--temperature=1.0 \
--output_length=200 \
--starter="In the Second Age of Middle-earth, the lords of Elves, Dwarves, and Men are given Rings of Power. Unbeknownst to them, the Dark Lord Sauron forges the One Ring in Mount Doom, infusing into it a great part of his power to dominate, through it and at a distance, the other Rings, so he might conquer Middle-earth. A final alliance of men and elves battles Sauron\'s forces in Mordor, where Prince Isildur of Gondor severs Sauron\'s finger, and the Ring with it, thereby destroying his physical form. With Sauron\'s first defeat, the Third Age of Middle-earth begins. Unfortunately, the Ring\'s influence corrupts Isildur, and, rather than destroy the Ring, Isildur takes it for himself. Isildur is later killed by Orcs, and the Ring is lost for 2,500 years, until it is found by Gollum, who owns it for five centuries. The Ring is then found by a hobbit named Bilbo Baggins, who turns invisible when he puts it on, but is unaware of its history. TL;DR:"
```

However, our experience of the output is somewhat disappointing. For example, these are a couple of "summaries" generated from the above LOTR text:

```
TL;DR: far darker than we know, one of the oldest, largest tongue- ancient and vast in history even. 
The encirzation stars symbol, thought to be any rectangles, scrimshaw, or awkwardly spaced (less variohed to).be issued by our language. 
Its making power achievable by all through magic consumption. identity is the consuming power from sacrificial outpourings. 
Rivendell entertainment; stunt-star beford bounty awarded its Moron. against. Anyone may enables its,'production. 
Treason involves.' but He speculated whon, if was this power utilized. it goes as if he, than unfairly. 
around culture goes by Danny Moby had Pergola ate mystery haunceaced Tickets Yes
```  

```
TL;DR: after the adventure designs Lost and found the would, in Baggins & Gimli & Galadriel will go there first and only Sulando the housebonder will, 
run from there when the 3sus take Telemachus and not. If absorbed by, is he unaware by the 4sing evil in his form at. 
Or Barliman thepotent only, wolves v\ a danger. Terror Axits his lives,d { dont confissions even rhodes char Youoglas, 
through onin he himself demands 7 with it to. 1861 would seven hoursa in which they would an out is going to fight speedtic happenspses there. 
theirs Put'sunder where some always been Since the days to know not Known bear into dyes anymore. prior disclose knowledge Knowing of the Lies. 
The key' lies and arrayed. It, thereafter of thingmouth yet refuses, will endure Miracles up without spelling Lem. lesions and roots.
```

As you can see, the results are too long, non-factual, and only vaguely related to the starter. They are more like text generation instead of summarization. Notice that the generated text has some key important LOTR concepts that do not exist in the starter text, such as "Rivendell" (an elf city), "Gimli" (a dwarf character), "Galadriel" (an elf character) and etc. This indicates the original training data have articles about LOTR, and the pre-trained model actually memorized these concepts, and pull them out during the inference (probably through the attention mechanism)

Next, let's see if the performance can be improved by fine-tuning. We use the [cnn-dailymail](https://cs.nyu.edu/~kcho/DMQA/) dataset, which has 290k news articles and a few "highlights" for each article. We create the ground-truth training data by concatenating each article with one of its highlights (randomly picked). Here are some examples:

___Ground Truth One___

```
If you turn to the Bible -- Isaiah Chapter 35, Verse 8 -- you will see a passage that in part says, "A highway shall be there, and a road, and it shall be called the Highway of Holiness."

Churchgoers in six states have held prayer sessions along the side of Interstate 35.

Now, is it possible that this "highway" mentioned in Chapter 35 is actually Interstate 35 that runs through six U.S. states, from southern Texas to northern Minnesota? Some Christians have faith that is indeed the case.

... Truncated for brevity ...

But on the side of the road, the prayerful aren't going to change their minds. Holy highways and nude clubs, they believe, are not a combination God has in mind. E-mail to a friend
TL;DR:


I-35 runs from southern Texas to northern Minnesota<|endoftext|>

```

___Ground Truth Two___

```
(InStyle) -- It all boils down to this. It doesn't really matter all that much what hot, nubile French maverick has set the fashion world on fire. Or which Milanese visionary has a new fabric technique discovered during a life-changing trip to Angkor Wat that's sure to bring back sixties minimalism with a twist. Or that so-and-so has signed a deal to develop boutique spa hotels around the globe in former monasteries. Because, in the end, he's Ralph Lauren, and we're not.

Ralph Lauren has his eye on China and Japan.

... Truncated for brevity ...


Get a FREE TRIAL issue of InStyle - CLICK HERE!

Copyright © 2007 Time Inc. All rights reserved.
TL;DR:


Ralph Lauren began as tie salesman from the Bronx
```

To fine-tune the `355M` model, we point the `dataset_path` to the [preprocessed cnn-dailymail dataset](#data) and specify `cnndm` as the loader. Here we fine-tune the model for one epoch with 2000 steps:

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
--base_lr=0.0001 \
--num_epoch=1 \
--steps_per_epoch=2000
```

Here are some summaries from the fine-tuned model. They are significantly better than the pre-trained model: the results are much more concise and associated with the starter text.

___Fine-tuned Example One___

```
In the Second Age of Middle-earth, the lords of Elves, Dwarves, and Men are given Rings of Power. Unbeknownst to them, 
the Dark Lord Sauron forges the One Ring in Mount Doom, infusing into it a great part of his power to dominate, 
through it and at a distance, the other Rings, so he might conquer Middle-earth. A final alliance of men and elves battles Sauron's forces in Mordor, 
where Prince Isildur of Gondor severs Sauron's finger, and the Ring with it, thereby destroying his physical form. With Sauron's first defeat, 
the Third Age of Middle-earth begins. Unfortunately, the Ring's influence corrupts Isildur, and, rather than destroy the Ring, 
Isildur takes it for himself. Isildur is later killed by Orcs, and the Ring is lost for 2,500 years, until it is found by Gollum, 
who owns it for five centuries. The Ring is then found by a hobbit named Bilbo Baggins, who turns invisible when he puts it on, 
but is unaware of its history.
TL;DR:

# Result 1
The Ring is believed to have been lost primarily for 25 years


# Result 2
Excess Ring content contributed to Final Age of Middle-earth

# Result 3
The ring is found by Gollum
```

___Fine-tuned Example Two___

```
TensorFlow [1] is an interface for expressing machine learning algorithms, and an implementation for executing such algorithms. 
A computation expressed using TensorFlow can be executed with little or no change on a wide variety of heterogeneous systems, 
ranging from mobile devices such as phones and tablets up to large-scale distributed systems of hundreds of machines and 
thousands of computational devices such as GPU cards. The system is flexible and can be used to express a wide variety of algorithms, 
including training and inference algorithms for deep neural network models, and it has been used for conducting research and for deploying 
machine learning systems into production across more than a dozen areas of computer science and other fields, including speech recognition, 
computer vision, robotics, information retrieval, natural language processing, geographic information extraction, 
and computational drug discovery. This paper describes the TensorFlow interface and an implementation of that interface that we 
have built at Google. The TensorFlow API and a reference implementation were released as an open-source package under the 
Apache 2.0 license in November, 2015 and are available at www.tensorflow.org.
TL;DR:

# Result 1
TensorFlow software was a section of Google's foundation software suite

# Result 2
(1) TensorFlow interface for machine learning algorithms and an implementation of that interface

# Result 3
TensorFlow was built to express computer learning processes
```


### Conversational Question And Answering <a name="conversational-qa"></a>

Conversational Question Answering is the ability to understand a text passage and answer a series of interconnected questions that appear in a conversation. We can also treat it as a conditional text generation problem: the condition (starter) is the text passage plus the start of a conversation, the task is to continue the conversation by asking new questions about the text passage and delivering convincing answers.

For example, we can use a text passage about 2008 Summer Olympics torch relay and a few questions and answer as the starter:

```
The 2008 Summer Olympics torch relay was run from March 24 until August 8, 2008, prior to the 2008 Summer Olympics, with the theme of “one world, one dream”. Plans for the relay were announced on April 26, 2007, in Beijing, China. The relay, also called by the organizers as the “Journey of Harmony”, lasted 129 days and carried the torch 137,000 km (85,000 mi) – the longest distance of any Olympic torch relay since the tradition was started ahead of the 1936 Summer Olympics. After being lit at the birthplace of the Olympic Games in Olympia, Greece on March 24, the torch traveled to the Panathinaiko Stadium in Athens, and then to Beijing, arriving on March 31. From Beijing, the torch was following a route passing through six continents. The torch has visited cities along the Silk Road, symbolizing ancient links between China and the rest of the world. The relay also included an ascent with the flame to the top of Mount Everest on the border of Nepal and Tibet, China from the Chinese side, which was closed specially for the event. 
 Q: What was the theme? 
 A: “one world, one dream”. 
 Q: What was the length of the race? 
 A: 137,000 km 
 Q: Was it larger than previous ones? 
 A: No 
 Q: Where did the race begin? 
 A: Olympia, Greece 
 Q: Is there anything notable about that place? 
 A: birthplace of Olympic Games 
 Q: Where did they go after? 
 A: Athens 
 Q: How many days was the race? 
 A: seven 
 Q: Did they visit any notable landmarks? 
 A: Panathinaiko Stadium 
 Q: And did they climb any mountains?
```

Let's first see the performance of the pre-trained GPT2 model (355M) without fine-tuning. Here is the generated conversation:

```
 A: Saturry 
 Q: ? Where did they go? A: Colorado Rockies
  Q: had many landmarks, a mountain peak
 also similar finish to in Pad about 20 miles ... one, a human – small parachute.

Q of demanding Def Dr Michelle Row, Paradise Valley located would be towns.

A hike (: home week period dress or shoot ?

Ryan Kerrigan Q: they from Philly? and LA en route?

a: Philadelphia A Department in Louisa is connected to there city Sydney Walker taking such suburb. it is now planned into.

It particular Edmonton like our number 5

The P hunting action move an herb Bruce head with rapresined me New king in incoming by wrIST references Lincoln Via paper or altogether living remake Hanged Henry Colorado Arrows Denver from San Gerhardt to Alemany. Eugene who Brady by working Six many the Serpent to Ari who entire Wit's blessing Trinity from W and Tennessee
```

One can see the generated text is not a convincing conversation: it quickly drifted away from the rhythm of a conversation, neither the questions and answers are valid, and the output does reflect the context of the text passage very well.


To fine-tune the model, we use the [CoQA dataset](https://stanfordnlp.github.io/coqa/), which contains 8000+ texts (conversations) and 127000+ questions, that is about 15.875 questions per text (conversation). This is an example from the ground-truth conversation. 


```
Teary Joe was a boy with a special ability: he could make himself cry in less than a second. If he disliked something, or things became difficult, Teary Joe would not hesitate to put on a pitiful face and set great big tears running down his cheeks. In this way he managed to get practically everything he wanted, because no one could resist the pity inspired by his tearful little face. 

... Truncated for brevity ...


In the end there was no cake. But that wasn't so bad, because Joe discovered it had been much more fun doing all those things that evening rather than just sitting crying to get a piece of cake that, in the end, wouldn't have been worth it.
 Q: Where did Teary and Pipo meet?
 A: Pipo was asking for change
 Q: Where?
 A: in the street

... Truncated for brevity ...

 Q: When he got home what did he want?
 A: cake
 Q: Did he get it?
 A: no
```


The following command fine-tunes the 355M model for one epoch with 2000 steps on the `CoQA` dataset:


```
python finetune.py \
--model=355M \
--model_ckpt=models/355M/model.ckpt \
--json_hparams=models/355M/hparams.json \
--json_encoder=models/355M/encoder.json \
--vocab_bpe=models/355M/vocab.bpe \
--output_name=coqa_355M_1x2000.h5 \
--dataset_path=/home/ubuntu/data/coqa \
--data_loader=coqa \
--base_lr=0.0001 \
--num_epoch=1 \
--steps_per_epoch=2000
```

And here is the command that uses the fine-tuned model to generate conversation based on the above 2008 Summer Olympics torch relay article:

```
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



# End of the starter
 Q: And did they climb any mountains?

# Generated conversation 
 A: Yes  Mount Everest
 Q: What did they hope to reach?
 A: Top of mountain Everest, border of Nepal and Tibet
 Q: Where did they plan to go next?
 A: Beijing, China.
 Q: What concern did competition over profits?
 A: train accident
 Q: On what day, did the gold start losing value?
 A: March 24
 Q: What was the theme for the 2009 Olympics?
 A: journey of Harmony,
 Q: Was that a solar event?
 A: Yes Rawl083
```

The fine-tuned model produced text in a better format, contextually closely related to the starter and logically valid questions and answers. However, the conversation is still not flawless. For example, there should be no Olympics in the year 2009, and a "solar event" named "Rawl083" should only exist in science fiction.


## Small Models <a name="small-models"></a>

### Text Generation 124M <a name="text-generation-124M"></a>

```
python finetune.py \
--model=124M \
--model_ckpt=models/124M/model.ckpt \
--json_hparams=models/124M/hparams.json \
--json_encoder=models/124M/encoder.json \
--vocab_bpe=models/124M/vocab.bpe \
--output_name=killbill_124M_1x2000.h5 \
--dataset_path=dataset/killbill.txt \
--data_loader=text \
--base_lr=0.0001 \
--num_epoch=1 \
--steps_per_epoch=2000

python inference.py \
--model_path=output/killbill_124M_1x2000.h5 \
--json_hparams=models/124M/hparams.json \
--json_encoder=models/124M/encoder.json \
--vocab_bpe=models/124M/vocab.bpe \
--nucleus \
--top_p=1.0 \
--temperature=1.0 \
--output_length=200 \
--batch_size=5 \
--starter='She picked up the sword'

```

```
python finetune.py \
--model=124M \
--model_ckpt=models/124M/model.ckpt \
--json_hparams=models/124M/hparams.json \
--json_encoder=models/124M/encoder.json \
--vocab_bpe=models/124M/vocab.bpe \
--output_name=CompleteRRMartin_124M_1x2000.h5 \
--dataset_path=dataset/CompleteRRMartin.txt \
--data_loader=text \
--base_lr=0.0001 \
--num_epoch=1 \
--steps_per_epoch=2000

python inference.py \
--model_path=output/CompleteRRMartin_124M_1x2000.h5 \
--json_hparams=models/124M/hparams.json \
--json_encoder=models/124M/encoder.json \
--vocab_bpe=models/124M/vocab.bpe \
--nucleus \
--top_p=1.0 \
--temperature=1.0 \
--output_length=200 \
--batch_size=5 \
--starter='She picked up the sword'
```

### Text Summarization 124M <a name="text-summarization-124M"></a>

```
python finetune.py \
--model=124M \
--model_ckpt=models/124M/model.ckpt \
--json_hparams=models/124M/hparams.json \
--json_encoder=models/124M/encoder.json \
--vocab_bpe=models/124M/vocab.bpe \
--output_name=cnndm_124M_1x2000.h5 \
--dataset_path=/home/ubuntu/data/summarization \
--data_loader=cnndm \
--base_lr=0.0001 \
--num_epoch=1 \
--steps_per_epoch=2000


python inference.py \
--model_path=output/cnndm_124M_1x2000.h5 \
--json_hparams=models/124M/hparams.json \
--json_encoder=models/124M/encoder.json \
--vocab_bpe=models/124M/vocab.bpe \
--nucleus \
--top_p=1.0 \
--temperature=1.0 \
--output_length=200 \
--batch_size=5 \
--starter="In the Second Age of Middle-earth, the lords of Elves, Dwarves, and Men are given Rings of Power. Unbeknownst to them, the Dark Lord Sauron forges the One Ring in Mount Doom, infusing into it a great part of his power to dominate, through it and at a distance, the other Rings, so he might conquer Middle-earth. A final alliance of men and elves battles Sauron\'s forces in Mordor, where Prince Isildur of Gondor severs Sauron\'s finger, and the Ring with it, thereby destroying his physical form. With Sauron\'s first defeat, the Third Age of Middle-earth begins. Unfortunately, the Ring\'s influence corrupts Isildur, and, rather than destroy the Ring, Isildur takes it for himself. Isildur is later killed by Orcs, and the Ring is lost for 2,500 years, until it is found by Gollum, who owns it for five centuries. The Ring is then found by a hobbit named Bilbo Baggins, who turns invisible when he puts it on, but is unaware of its history. \nTL;DR:\n"

python inference.py \
--model_path=output/cnndm_124M_1x2000.h5 \
--json_hparams=models/124M/hparams.json \
--json_encoder=models/124M/encoder.json \
--vocab_bpe=models/124M/vocab.bpe \
--nucleus \
--top_p=1.0 \
--temperature=1.0 \
--output_length=200 \
--batch_size=5 \
--starter="TensorFlow [1] is an interface for expressing machine learning algorithms, and an implementation for executing such algorithms. A computation expressed using TensorFlow can be executed with little or no change on a wide variety of heterogeneous systems, ranging from mobile devices such as phones and tablets up to large-scale distributed systems of hundreds of machines and thousands of computational devices such as GPU cards. The system is flexible and can be used to express a wide variety of algorithms, including training and inference algorithms for deep neural network models, and it has been used for conducting research and for deploying machine learning systems into production across more than a dozen areas of computer science and other fields, including speech recognition, computer vision, robotics, information retrieval, natural language processing, geographic information extraction, and computational drug discovery. This paper describes the TensorFlow interface and an implementation of that interface that we have built at Google. The TensorFlow API and a reference implementation were released as an open-source package under the Apache 2.0 license in November, 2015 and are available at www.tensorflow.org. \nTL;DR:\n"
```

### Conversational Question And Answering 124M <a name="conversational-qa-124M"></a>

```
python finetune.py \
--model=124M \
--model_ckpt=models/124M/model.ckpt \
--json_hparams=models/124M/hparams.json \
--json_encoder=models/124M/encoder.json \
--vocab_bpe=models/124M/vocab.bpe \
--output_name=coqa_124M_1x2000.h5 \
--dataset_path=/home/ubuntu/data/coqa \
--data_loader=coqa \
--base_lr=0.0001 \
--num_epoch=1 \
--steps_per_epoch=2000

python inference.py \
--model_path=output/coqa_124M_1x2000.h5 \
--json_hparams=models/124M/hparams.json \
--json_encoder=models/124M/encoder.json \
--vocab_bpe=models/124M/vocab.bpe \
--nucleus \
--top_p=1.0 \
--temperature=1.0 \
--output_length=200 \
--batch_size=5 \
--starter="The 2008 Summer Olympics torch relay was run from March 24 until August 8, 2008, prior to the 2008 Summer Olympics, with the theme of \“one world, one dream\”. Plans for the relay were announced on April 26, 2007, in Beijing, China. The relay, also called by the organizers as the “Journey of Harmony”, lasted 129 days and carried the torch 137,000 km (85,000 mi) – the longest distance of any Olympic torch relay since the tradition was started ahead of the 1936 Summer Olympics. After being lit at the birthplace of the Olympic Games in Olympia, Greece on March 24, the torch traveled to the Panathinaiko Stadium in Athens, and then to Beijing, arriving on March 31. From Beijing, the torch was following a route passing through six continents. The torch has visited cities along the Silk Road, symbolizing ancient links between China and the rest of the world. The relay also included an ascent with the flame to the top of Mount Everest on the border of Nepal and Tibet, China from the Chinese side, which was closed specially for the event. \n Q: What was the theme? \n A: “one world, one dream”. \n Q: What was the length of the race? \n A: 137,000 km \n Q: Was it larger than previous ones? \n A: No \n Q: Where did the race begin? \n A: Olympia, Greece \n Q: Is there anything notable about that place? \n A: birthplace of Olympic Games \n Q: Where did they go after? \n A: Athens \n Q: How many days was the race? \n A: seven \n Q: Did they visit any notable landmarks? \n A: Panathinaiko Stadium \n Q: And did they climb any mountains? \n A:"
```

## Medium Models <a name="medium-models"></a>

### Text Generation 355M <a name="text-generation-355M"></a>

```
python finetune.py \
--model=355M \
--model_ckpt=models/355M/model.ckpt \
--json_hparams=models/355M/hparams.json \
--json_encoder=models/355M/encoder.json \
--vocab_bpe=models/355M/vocab.bpe \
--output_name=killbill_355M_1x2000.h5 \
--dataset_path=dataset/killbill.txt \
--data_loader=text \
--base_lr=0.0001 \
--num_epoch=1 \
--steps_per_epoch=2000

python inference.py \
--model_path=output/killbill_355M_1x2000.h5 \
--json_hparams=models/355M/hparams.json \
--json_encoder=models/355M/encoder.json \
--vocab_bpe=models/355M/vocab.bpe \
--nucleus \
--top_p=1.0 \
--temperature=1.0 \
--output_length=200 \
--batch_size=5 \
--starter='She picked up the sword'
```

```
python finetune.py \
--model=355M \
--model_ckpt=models/355M/model.ckpt \
--json_hparams=models/355M/hparams.json \
--json_encoder=models/355M/encoder.json \
--vocab_bpe=models/355M/vocab.bpe \
--output_name=CompleteRRMartin_355M_1x2000.h5 \
--dataset_path=dataset/CompleteRRMartin.txt \
--data_loader=text \
--base_lr=0.0001 \
--num_epoch=1 \
--steps_per_epoch=2000

python inference.py \
--model_path=output/CompleteRRMartin_355M_1x2000.h5 \
--json_hparams=models/355M/hparams.json \
--json_encoder=models/355M/encoder.json \
--vocab_bpe=models/355M/vocab.bpe \
--nucleus \
--top_p=1.0 \
--temperature=1.0 \
--output_length=200 \
--batch_size=5 \
--starter='She picked up the sword'
```

### Text Summarization 355M <a name="text-summarization-355M"></a>

```
python finetune.py \
--model=355M \
--model_ckpt=models/355M/model.ckpt \
--json_hparams=models/355M/hparams.json \
--json_encoder=models/355M/encoder.json \
--vocab_bpe=models/355M/vocab.bpe \
--output_name=cnndm_355M_1x2000.h5 \
--dataset_path=/home/ubuntu/data/summarization \
--data_loader=cnndm \
--base_lr=0.0001 \
--num_epoch=1 \
--steps_per_epoch=2000


python inference.py \
--model_path=output/cnndm_355M_1x2000.h5 \
--json_hparams=models/355M/hparams.json \
--json_encoder=models/355M/encoder.json \
--vocab_bpe=models/355M/vocab.bpe \
--nucleus \
--top_p=1.0 \
--temperature=1.0 \
--output_length=200 \
--batch_size=5 \
--starter="In the Second Age of Middle-earth, the lords of Elves, Dwarves, and Men are given Rings of Power. Unbeknownst to them, the Dark Lord Sauron forges the One Ring in Mount Doom, infusing into it a great part of his power to dominate, through it and at a distance, the other Rings, so he might conquer Middle-earth. A final alliance of men and elves battles Sauron\'s forces in Mordor, where Prince Isildur of Gondor severs Sauron\'s finger, and the Ring with it, thereby destroying his physical form. With Sauron\'s first defeat, the Third Age of Middle-earth begins. Unfortunately, the Ring\'s influence corrupts Isildur, and, rather than destroy the Ring, Isildur takes it for himself. Isildur is later killed by Orcs, and the Ring is lost for 2,500 years, until it is found by Gollum, who owns it for five centuries. The Ring is then found by a hobbit named Bilbo Baggins, who turns invisible when he puts it on, but is unaware of its history. \nTL;DR:\n"

python inference.py \
--model_path=output/cnndm_355M_1x2000.h5 \
--json_hparams=models/355M/hparams.json \
--json_encoder=models/355M/encoder.json \
--vocab_bpe=models/355M/vocab.bpe \
--nucleus \
--top_p=1.0 \
--temperature=1.0 \
--output_length=200 \
--batch_size=5 \
--starter="TensorFlow [1] is an interface for expressing machine learning algorithms, and an implementation for executing such algorithms. A computation expressed using TensorFlow can be executed with little or no change on a wide variety of heterogeneous systems, ranging from mobile devices such as phones and tablets up to large-scale distributed systems of hundreds of machines and thousands of computational devices such as GPU cards. The system is flexible and can be used to express a wide variety of algorithms, including training and inference algorithms for deep neural network models, and it has been used for conducting research and for deploying machine learning systems into production across more than a dozen areas of computer science and other fields, including speech recognition, computer vision, robotics, information retrieval, natural language processing, geographic information extraction, and computational drug discovery. This paper describes the TensorFlow interface and an implementation of that interface that we have built at Google. The TensorFlow API and a reference implementation were released as an open-source package under the Apache 2.0 license in November, 2015 and are available at www.tensorflow.org. \nTL;DR:\n"
```

### Conversational Question And Answering 355M <a name="conversational-qa-355M"></a>

```
python finetune.py \
--model=355M \
--model_ckpt=models/355M/model.ckpt \
--json_hparams=models/355M/hparams.json \
--json_encoder=models/355M/encoder.json \
--vocab_bpe=models/355M/vocab.bpe \
--output_name=coqa_355M_1x2000.h5 \
--dataset_path=/home/ubuntu/data/coqa \
--data_loader=coqa \
--base_lr=0.0001 \
--num_epoch=1 \
--steps_per_epoch=2000

python inference.py \
--model_path=output/coqa_355M_1x2000.h5 \
--json_hparams=models/355M/hparams.json \
--json_encoder=models/355M/encoder.json \
--vocab_bpe=models/355M/vocab.bpe \
--nucleus \
--top_p=1.0 \
--temperature=1.0 \
--output_length=200 \
--batch_size=5 \
--starter="The 2008 Summer Olympics torch relay was run from March 24 until August 8, 2008, prior to the 2008 Summer Olympics, with the theme of \“one world, one dream\”. Plans for the relay were announced on April 26, 2007, in Beijing, China. The relay, also called by the organizers as the “Journey of Harmony”, lasted 129 days and carried the torch 137,000 km (85,000 mi) – the longest distance of any Olympic torch relay since the tradition was started ahead of the 1936 Summer Olympics. After being lit at the birthplace of the Olympic Games in Olympia, Greece on March 24, the torch traveled to the Panathinaiko Stadium in Athens, and then to Beijing, arriving on March 31. From Beijing, the torch was following a route passing through six continents. The torch has visited cities along the Silk Road, symbolizing ancient links between China and the rest of the world. The relay also included an ascent with the flame to the top of Mount Everest on the border of Nepal and Tibet, China from the Chinese side, which was closed specially for the event. \n Q: What was the theme? \n A: “one world, one dream”. \n Q: What was the length of the race? \n A: 137,000 km \n Q: Was it larger than previous ones? \n A: No \n Q: Where did the race begin? \n A: Olympia, Greece \n Q: Is there anything notable about that place? \n A: birthplace of Olympic Games \n Q: Where did they go after? \n A: Athens \n Q: How many days was the race? \n A: seven \n Q: Did they visit any notable landmarks? \n A: Panathinaiko Stadium \n Q: And did they climb any mountains? \n A:"
```


## Large Models <a name="large-models"></a>

### Text Generation 774M <a name="text-generation-774M"></a>

```
python finetune.py \
--model=774M \
--model_ckpt=models/774M/model.ckpt \
--json_hparams=models/774M/hparams.json \
--json_encoder=models/774M/encoder.json \
--vocab_bpe=models/774M/vocab.bpe \
--output_name=killbill_774M_1x2000.h5 \
--dataset_path=dataset/killbill.txt \
--data_loader=text \
--base_lr=0.0001 \
--num_epoch=1 \
--steps_per_epoch=2000

python inference.py \
--model_path=output/killbill_774M_1x2000.h5 \
--json_hparams=models/774M/hparams.json \
--json_encoder=models/774M/encoder.json \
--vocab_bpe=models/774M/vocab.bpe \
--nucleus \
--top_p=1.0 \
--temperature=1.0 \
--output_length=200 \
--batch_size=5 \
--starter='She picked up the sword'
```

```
python finetune.py \
--model=774M \
--model_ckpt=models/774M/model.ckpt \
--json_hparams=models/774M/hparams.json \
--json_encoder=models/774M/encoder.json \
--vocab_bpe=models/774M/vocab.bpe \
--output_name=CompleteRRMartin_774M_1x2000.h5 \
--dataset_path=dataset/CompleteRRMartin.txt \
--data_loader=text \
--base_lr=0.0001 \
--num_epoch=1 \
--steps_per_epoch=2000

python inference.py \
--model_path=output/CompleteRRMartin_774M_1x2000.h5 \
--json_hparams=models/774M/hparams.json \
--json_encoder=models/774M/encoder.json \
--vocab_bpe=models/774M/vocab.bpe \
--nucleus \
--top_p=1.0 \
--temperature=1.0 \
--output_length=200 \
--batch_size=5 \
--starter='She picked up the sword'
```

### Text Summarization 774M <a name="text-summarization-774M"></a>

```
python finetune.py \
--model=774M \
--model_ckpt=models/774M/model.ckpt \
--json_hparams=models/774M/hparams.json \
--json_encoder=models/774M/encoder.json \
--vocab_bpe=models/774M/vocab.bpe \
--output_name=cnndm_774M_1x2000.h5 \
--dataset_path=/home/ubuntu/data/summarization \
--data_loader=cnndm \
--base_lr=0.0001 \
--num_epoch=1 \
--steps_per_epoch=2000


python inference.py \
--model_path=output/cnndm_774M_1x2000.h5 \
--json_hparams=models/774M/hparams.json \
--json_encoder=models/774M/encoder.json \
--vocab_bpe=models/774M/vocab.bpe \
--nucleus \
--top_p=1.0 \
--temperature=1.0 \
--output_length=200 \
--batch_size=5 \
--starter="In the Second Age of Middle-earth, the lords of Elves, Dwarves, and Men are given Rings of Power. Unbeknownst to them, the Dark Lord Sauron forges the One Ring in Mount Doom, infusing into it a great part of his power to dominate, through it and at a distance, the other Rings, so he might conquer Middle-earth. A final alliance of men and elves battles Sauron\'s forces in Mordor, where Prince Isildur of Gondor severs Sauron\'s finger, and the Ring with it, thereby destroying his physical form. With Sauron\'s first defeat, the Third Age of Middle-earth begins. Unfortunately, the Ring\'s influence corrupts Isildur, and, rather than destroy the Ring, Isildur takes it for himself. Isildur is later killed by Orcs, and the Ring is lost for 2,500 years, until it is found by Gollum, who owns it for five centuries. The Ring is then found by a hobbit named Bilbo Baggins, who turns invisible when he puts it on, but is unaware of its history. \nTL;DR:\n"

python inference.py \
--model_path=output/cnndm_774M_1x2000.h5 \
--json_hparams=models/774M/hparams.json \
--json_encoder=models/774M/encoder.json \
--vocab_bpe=models/774M/vocab.bpe \
--nucleus \
--top_p=1.0 \
--temperature=1.0 \
--output_length=200 \
--batch_size=5 \
--starter="TensorFlow [1] is an interface for expressing machine learning algorithms, and an implementation for executing such algorithms. A computation expressed using TensorFlow can be executed with little or no change on a wide variety of heterogeneous systems, ranging from mobile devices such as phones and tablets up to large-scale distributed systems of hundreds of machines and thousands of computational devices such as GPU cards. The system is flexible and can be used to express a wide variety of algorithms, including training and inference algorithms for deep neural network models, and it has been used for conducting research and for deploying machine learning systems into production across more than a dozen areas of computer science and other fields, including speech recognition, computer vision, robotics, information retrieval, natural language processing, geographic information extraction, and computational drug discovery. This paper describes the TensorFlow interface and an implementation of that interface that we have built at Google. The TensorFlow API and a reference implementation were released as an open-source package under the Apache 2.0 license in November, 2015 and are available at www.tensorflow.org. \nTL;DR:\n"
```

### Conversational Question And Answering 774M <a name="conversational-qa-774M"></a>

```
python finetune.py \
--model=774M \
--model_ckpt=models/774M/model.ckpt \
--json_hparams=models/774M/hparams.json \
--json_encoder=models/774M/encoder.json \
--vocab_bpe=models/774M/vocab.bpe \
--output_name=coqa_774M_1x2000.h5 \
--dataset_path=/home/ubuntu/data/coqa \
--data_loader=coqa \
--base_lr=0.0001 \
--num_epoch=1 \
--steps_per_epoch=2000

python inference.py \
--model_path=output/coqa_774M_1x2000.h5 \
--json_hparams=models/774M/hparams.json \
--json_encoder=models/774M/encoder.json \
--vocab_bpe=models/774M/vocab.bpe \
--nucleus \
--top_p=1.0 \
--temperature=1.0 \
--output_length=200 \
--batch_size=5 \
--starter="The 2008 Summer Olympics torch relay was run from March 24 until August 8, 2008, prior to the 2008 Summer Olympics, with the theme of \“one world, one dream\”. Plans for the relay were announced on April 26, 2007, in Beijing, China. The relay, also called by the organizers as the “Journey of Harmony”, lasted 129 days and carried the torch 137,000 km (85,000 mi) – the longest distance of any Olympic torch relay since the tradition was started ahead of the 1936 Summer Olympics. After being lit at the birthplace of the Olympic Games in Olympia, Greece on March 24, the torch traveled to the Panathinaiko Stadium in Athens, and then to Beijing, arriving on March 31. From Beijing, the torch was following a route passing through six continents. The torch has visited cities along the Silk Road, symbolizing ancient links between China and the rest of the world. The relay also included an ascent with the flame to the top of Mount Everest on the border of Nepal and Tibet, China from the Chinese side, which was closed specially for the event. \n Q: What was the theme? \n A: “one world, one dream”. \n Q: What was the length of the race? \n A: 137,000 km \n Q: Was it larger than previous ones? \n A: No \n Q: Where did the race begin? \n A: Olympia, Greece \n Q: Is there anything notable about that place? \n A: birthplace of Olympic Games \n Q: Where did they go after? \n A: Athens \n Q: How many days was the race? \n A: seven \n Q: Did they visit any notable landmarks? \n A: Panathinaiko Stadium \n Q: And did they climb any mountains? \n A:"
```

## Evaluation <a name="evaluation"></a>


### Text Summarization (CNNDM) <a name="evaluation-cnndm"></a>

We follow the instructions in this [link](http://opennmt.net/OpenNMT-py/Summarization.html) to evaluate the models for text summization.

__Step One__: Download the test split from this [link](https://github.com/harvardnlp/sent-summary).


__Step Two__: Generate prediction output. For example, the command below generates the results for the fine-tuned 124M model.

```
python evaluate.py \
--model_path=output/cnndm_124M_1x2000.h5 \
--json_hparams=models/124M/hparams.json \
--json_encoder=models/124M/encoder.json \
--vocab_bpe=models/124M/vocab.bpe \
--dataset_path=/home/ubuntu/data/summarization \
--data_loader=cnndm \
--nucleus \
--top_p=1.0 \
--temperature=1.0 \
--output_length=100 \
--output_file=results/cnndm_124M_1x2000_test.output
```

__Step Three__: Install ROUGE package and the evaluation script. We adopt the instructions on this [blog](https://poojithansl7.wordpress.com/2018/08/04/setting-up-rouge/) to work with this [repo](https://github.com/sebastianGehrmann/rouge-baselines.git).

```
sudo apt-get update
sudo apt-get install perl
sudo apt-get install synaptic
# Use synaptic to install libxml-dom-perl

git clone https://github.com/sebastianGehrmann/rouge-baselines.git --recursive
cd rouge-baselines
virtualenv -p /usr/bin/python3.6 venv
. venv/bin/activate

cd pyrouge
python setup.py build
python setup.py install

pyrouge_set_rouge_path path_to/RELEASE-1.5.5

cd RELEASE-1.5.5/data/WordNet-2.0-Exceptions
./buildExeptionDB.pl . exc WordNet-2.0.exc.db
cd ..
ln -s WordNet-2.0-Exceptions/WordNet-2.0.exc.db WordNet-2.0.exc.db

# Test
cd ../../
python -m pyrouge.test
```

__Step Four__: Run evaluation.

```
cd rouge-baselines
. venv/bin/activate
python baseline.py -s path_to/prediction_results.out -t path_to/test.txt.tgt.tagged -m sent_tag_verbatim -r
```