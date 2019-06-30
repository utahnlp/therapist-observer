# Scripts

All the bash scripts, to set up experiments, consisted of 4 folders:
    1. "commands", all the commands to lanuch the preprocessing, training, evaltion are placed here, most of them will take a config file path in the "configs" folder as argument.
```
  <COMMAND>  <CONFIG_NAME>.sh
```
    2. "configs", all config files are stored here, in our cases, we organize them into "categorization" and "anticipation"
    3. "sbatchs", slurm scheduling scripts for remote server running, which may not be useful for your cases. Please either ignore that or porting to your own scheduler.
    4. "utils", a set of automative tools for easily adding or remove a configureation to both commands files and configs files, e.g. newly added hyperparameters, components. It will help you to prepare a new receipt for your new model. Then as mentioned in 1, delicious models will be easily built by feeding those configs files(reciepts) into the corresponding commands(cook tools).

In this document, we mainly introduce commands, configs and utils to give a guideline for easily exploring our code and por them to other tasks.

## Commands

All the following scripts can be a command, all of them are located in the commands folder. Every command require to take one argument, which is the path of a configuration file.
P.S. if you want to use relative path for the configuration file, be sure that you are invoking the `command` from the commands folder, other wise, you must use the absolute path for the configure file.


### Data Transformation (**trans.sh**)
    Orginal dataset is noisy and not quite convinient to use:
    1. It is a flatten csv format with one utterance per line and each line contains too many columns. Utterance in the same dialogue actually share many columns, however, for csv format, they are duplicated in each line for the same dialogue. What's more, It is not easy for reading like an interactive dialogue.
    2. Multilple selected annotation from different users existed for the same dialogue. However, some selected annotations are disagreed with each other. For example, there are are based on different utterance segmentation; Mispelled labels exist;
    3. It is based on original 28 MISC labels, label clustering are required.
    4. Original dialogue has more than 500+ utterance, which is too long so that it cannot be easily modeled. A dialogue sliding window are required.

    Hence, we normalize the dataset with the following transformation:
    1. transform flatten csv data with hierarchical json format
    2. Normalizing the disagreement on utterance segmentation and other labeling exceptions.
    3. Use label clustering strategy for rare labels.
    4. Segment the dialgue with a sliding context window.


### Data Placement (**place_data.sh**)

    `place_data.sh` will flatten all the json files into line-by-line json format, which will *save both time and memory for later preprocessing*. All the raw_data in the ```download``` folder will be flattened and copied into the ```psyc_ro``` folder, and then those files will be processed by tokenizing , preprocessing, prepare and other commands in the downstream commands.



### Tokenization (**tok.sh**)

    `tok.sh` will generate the tokenized sentence with spacy and write down into a new file. The setting for the input file and output file are in the required argument of a configuration file path. For psyc dataset, the untokenized sentences are in a "utterance" key of the json dict. This command will add a new key-value pair in the json dict. Key is "tokenized_utterance" and a list of tokens is its value  Hence, every original 'utterance' sentence will have a new tokenized token list in the output json. In later phases, only the tokenized data will be used. the original input data is read-only, and stored only for reference.
    When tokenizing, it will genenrate both the standard spacy tokenization, but also it will genenrate all the normalized tokenizations according to the rules in the special case dictionary
    After the tokenization, there will be a new folder genenerated at the side of the original read-only dataset folder. The new folder can be configured by "DATA_DIR" variable in the input configuration file .

```
You only need to do one time for tokenization, all the tokenized files will be genenerated in the DATA_DIR.
If those tokenized files has already existed in the DATA_DIR, then the tokenization will be skipped.
Hence, please clean the tokenization files you want to update its tokenization.
```

### Training Data Preparation (**preprocess_dataset.sh**)

    ```preprocess_dataset``` will take the above tokenized files as input and apply different strategies for preprocessing, for example, ```rmstop``` indicate whether to remove the stop words, and ```replace``` means how to replace the special entity, we use "no" for it.








