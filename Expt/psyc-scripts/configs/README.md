# Configs
    Usually, in ml experiments, different dataset or different parameters are two main group of setting.
    Hence, we write down all the required configurations in the seperate configuration files for different experiments. When running any other scripts, you must specifiy the path for a configuration file as the first argument for those scripts, such as ```./train.sh ./dev.sh ./dev_on_test.sh ./prepare.sh```
   The trained models will also been placed into corresponding folders seperately. So that we can iteratively run experiments different setting.
   `./add_args.sh` and `remove_args.sh` are designed to add or remove arguments to all the configuration files and all the command filesto make the arguments take effect.


    For each configiration, please refer `tensorflow/classes/config_reader.py` for details.



# Use Char Embedding

```
1. To use char emebeding, toggle the option `USE_CHAR_EMBED = x`
2. To training char embedding, toggle the option `TRAIN_CHAR_EMBEDDING = x`, training may add extra parameters slow the training,
3. To have an initial char embedding file, there is pretrained char embedding file `CHAR_EMB_FILE`
   Download emb file by execut the script in commands folder `./download_char_embedding.sh`
4. `CHAR_EMBED_SIZE=300` is the dimension of char embedding, when training the embeding, please to make it 100 or smaller one
5. We concat the final state of char bi-rnn to the origin word embedding, then the total dim of word_embeding is
   `WORD_EMBED_SIZE + 2 * HIDDEN_SIZE` 
6. How many character to compute an emebdding for the word. `MAX_CHAR_NUN_TO_KEEP`, for psyc dataset, 20 is enough.

# Training Elmo

For step 1 and step 2, we have two options to build graph.

```
1. use pretrained bilm model
Cons: dim too large, memory-consuming, too slow to run, usually need extra projection to concat with word emb
Pros: no need to train, it is also broad-coverage based on the wikipedia 

2. retraing the bilm
Cons: need retrain, require large corpus. also take 3 GPUs, around 20 hours , original model is 3 GPUS for 2 weeks.
Pros: All the better than the Cons for  pretrained bilm model
See more details on traning_elmo branchs and more details on FAQs
https://github.com/allenai/bilm-tf 
xxx_train.psyc.sh
xxx_ppl.sh
xxx_dumpwgt.sh

Worthy to mention, https://github.com/allenai/bilm-tf#whats-the-deal-with-n_characters-and-padding

```

## To test elmo

```
./preprocess_dataset_sc.sh
This script will using snt_dict to assign every snt with an id, which is the key for utterance cache.
All the files will be genenrated in the $RO_DATA/prep_data/xxxxx/


./prepare.sh ../configs/final_models/memnet_2hops_cma_gated_elmo_sc.sh
Prepare all the utterance cache , passage cache, question cache, and write them into the $VOCAB_DIR. (configured in the config file)
Check the prepare.log in the corresponding $WORK_DIR
For the whole psyc dataset, using pretrained model, it will take 40 hours

For 128 projection_dim and smaller cnn filters, it can be reduce to less than 15 hours.

./train.sh ../configs/final_models/memnet_2hops_cma_gated_elmo_sc.sh
train with the utterance cache , passage cache, question cache, and write them into the $VOCAB_DIR. (configured in the config file)
Check the training/train.log, see whether these cache are loaded.  (e.g Utterance cache loaded )

```
