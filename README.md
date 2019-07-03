<a href="#">
    <img src="https://www.mlciv.com/assets/img/therapist-observer2.png" alt="therapist logo" title="therapist observer" align="right" height="200" />
</a>

Therapist-Observer
==================

This repo implements a family of neural components for various hierarchical
dialogue models described in ["Observing Dialogue in Therapy:
Categorizing and Forcasting Behavioral Codes"](https://arxiv.org/pdf/1907.00326.pdf) By Cao et al. in
ACL 2019.
```
 @inproceedings{cao2019observing,
      author    = {Cao, Jie and Tanana, Michael and Imel, Zac E.
      and Poitras, Eric and Atkins, David C and Srikumar, Vivek},
      title     = {Observing Dialogue in Therapy: Categorizing and Forecasting Behavioral Codes},
      booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
      year      = {2019}
  }
```

Besides replicating the results on the psychotherapy dataset used in our
paper, we also offer a guideline or building models with the SOTA
neural components for conversational analysis in other domains.

# Table of Contents
*******************
<!--ts-->
   * [Therapist-Observer](#therapist-observer)
   * [Table of Contents](#table-of-contents)
   * [Part I. Usage](#part-i-usage)
      * [Required Software](#required-software)
      * [Data Preprocessing](#data-preprocessing)
      * [Preparing Embedding](#preparing-embedding)
      * [Training](#training)
         * [Training from scratch](#training-from-scratch)
         * [Analysis for training](#analysis-for-training)
         * [Resume Training from a checkpoint model](#resume-training-from-a-checkpoint-model)
      * [Evalution](#evalution)
   * [Part II. Experiment Desgining](#part-ii-experiment-desgining)
      * [Categorizing](#categorizing)
      * [Forecasting](#forecasting)
   * [Part VI. Usage for Other Dataset or Tasks](#part-vi-usage-for-other-dataset-or-tasks)
      * [Building Data Input](#building-data-input)
      * [Model Designing](#model-designing)
         * [Hierarchical Encoder](#hierarchical-encoder)
         * [Various Attention Mechansims](#various-attention-mechansims)
         * [Various Embeddings](#various-embeddings)
   * [Known Issues (To be moved to issues)](#known-issues-to-be-moved-to-issues)

<!-- Added by: jcao, at: Tue Jun  4 22:59:37 MDT 2019 -->

<!--te-->
# Part I. Usage
*******************

## Required Software

   - Install pyenv or other python environment manager

   In our case, we use pyenv and its plugin pyenv-virtualenv to set up
   the python environment. Please follow the detailed steps in
   https://github.com/pyenv/pyenv-virtualenv for details. Alternative
   environments management such as conda will be fine.

   - Install required packages

   ```bash
   pyenv install 2.7.12
   # in our default setting, we use `pyenv activate py2.7_tf1.4` to
   # activate the envivronment, please change this according to your preference.

   pyenv virtualenv 2.7 py2.7_tf1.4
   pyenv activate py2.7_tf1.4
   pip install tensorflow-gpu==1.4.0 spacy pandas ujson h5py sklearn matplotlib
   ```

   - Checkout this project.

   ```bash
       git clone git@github.com:utahnlp/therapist-observer.git therapist-observer
   ```
   `tensorflow` folder is the source code directory for nerual models.

   `Expt` folder is a folder for experiment managing, which includes all the commands(Expt/psyc_scripts/commands), config files(Expt/psyc_scripts/configs) to launch the experiments, and store all experiment outputs. In this repo, except `Expt/psyc_scirpts/commands/env.sh` contains the global variables, all model hyperparameters and reltaed configurations will be assigned in the config files in Expt/psyc_scripts/configs, each of them is corresponding to a model. For a detailed description for folders in `Expt` folder, please refer to [Expt README file](Expt/README.md)

## Data Preprocessing

Preprocessing pipeline consisted of 4 sub steps:
0) Put original data into `Expt/data/psyc_ro/download/data_filename` 
1) Data Transformation (**trans.sh**), check the path in `trans.sh` 
2) Dataset split and Placement (**place_data.sh**) 
3) Tokenization (**tok.sh**) 
4) Extra Preprocessing (**preprocess_dataset.sh**) 
The following command can run each of them in squeunce to fulfill the preprocessing pipeline.

```bash
# it will end after 30 minutes.
cd Expt/psyc-scripts/commands/
./pre_pipe.sh
```

When re-executing this, finished sub tasks will be skipped because the
correponding output folder has existed. Please manually delete the
corresponding folder for not skipping

For more details for preprocessing, please refer to document on [README of commands](Expt/psyc-scripts/commands/README.md)

## Preparing Embedding

```bash
# download glove.840B.300d into $RO_DATA_DIR,
# WORD_EMB_FILE in each config files will point to the path of this downloaded file
./download_glove.sh

# download elmo weights and options file into $DATA_DIR/psyc_elmo
# ELMO_OPTION_FILE and ELMO_WEIGHT_FILE will point the downloaded elmo weights and options file
./download_elmo.sh

# prepare vocabulary and elmo for training
# generating vocabulary embedding in $VOCAB_DIR in the corresponding config file
# which can be used by any task with $CONTEXT_WINDOW = 8, here, we take our selected model on categorizing client codes as a example.
./prepare.sh ../configs/categorizing/selected/C_C.sh

# Commands ends with "gpuid" means, CUDA_VISIBLEE_DEVICE will be specified by a second GPUID argument.
# ./prepare_gpuid.sh ../configs/categorizing/selected/C_C.sh 1
```

The above commands will mainly for preparing the vocabulary and building
elmo embeddings for every sentence and everytoken. When ELMo enabled,
this command may last for 25 minutes, and around 12G GPU memory.

You only need to do the preparation again when you need to update
the embeding, or you have retokenzied the data(token.sh), or you want
to build vocabulary for large context window. Once $VOCAB_DIR is
generated, this vocabulary can be used for other reciept by pointing
$VOCAB_DIR to this vocab folder.

All the following embedding related configurations in the config file
will impact the vocabulary preparation.

  - **WORD_EMB_FILE**

By default, we use glove.840B.300d, which is default value of WORD_EMB_FILE in our config files.
For using other word embedding, please change this configuration and do preparation again.

  - **ELMO_OPTION_FILE**, **ELMO_WEIGHT_FILE**

By default, these two files where point the default location of the download elmo files.
If using domain specific ELMo or other pretrained ELMo, make sure to change the above two variables in config file, and prepare.

 - **CONTEXT_WINDOW**

By simply set $CONTEXT_WINDOW=16, it is recommended to re-preprepare
the vocab when changing the window size.  Because when genenrating
sliding window dialogue segments, the words in last $CONTEXT_WINDOW
utterance of a dialogue may have slight impact on word frequency.

More details about the configuration, please refer to [README on configs](Expt/psyc-scripts/configs/README.md)

## Training

### Training from scratch
```bash
# all training command simply follows a single arguments
./train.sh <config_file>

# training from scratch, see `tensorflow/classes/config_reader.py` for details of each arguments in config_file
# Again, we use selected model on categoring client codes as an example, ../configs/categorizing/selected/C_C.sh
# $CONFIG_DIR will be made, train.log shows the training progress
# $CONFIG_DIR/models/ will save the models and checkpints every $STEPS_PER_CHECKPINTS batch
./train.sh ../configs/categorizing/selected/C_C.sh

# Commands ends with "gpuid" means, CUDA_VISIBLEE_DEVICE will be specified by a second GPUID argument.
./train_gpuid.sh ../configs/categorizing/selected/C_C.sh 1
```

Worth to mention, when training, best model with respect to different metric will be saved in $CONFIG_DIR/models/.
$CONFIG_DIR is required to be set in the model config file.

```
model prefix = $ALGO + sub_model_prefix.
```

$ALGO is just a name to identify your model. see `tensorflow/classes/config_reader.py` for more details.
$sub_model_prefix is relared to the metrics we used for evaluation, which follows a pattern "_A_B"

```
# A can be in {P, R, F1, R@K}
# B can be in {macro, weighted_macro, micro} and all MISC labels.
```

Hence, sub_model_prefix can be _F1_macro, that is what we used for our performance evaluation.

### Analysis for training
  ```bash
  # for analyzing training log for Patient(client) models
  python $ROOT_DIR/Expt/stats_scripts/stats_P.py train.log

  # for analyzing training log for Therapist models
  python $ROOT_DIR/Expt/stats_scripts/stats_T.py train.log
  ```

The whole training will last for around 20 hours on a V100 GPU. The following command will analyze the train.log and print current best performance.

### Resume Training from a checkpoint model

```bash
# training from saved checkpoint, matched by model file name with prefix as $MODEL_PREFIX_TO_RESTORE
./train_restore.sh <config_file> sub_model_prefix

# The sub_model_prefix argument is optional, when it is not loaded, the save model with best loss will be loaded. # However, model with smallest loss may not indicate best performance. You can resume from the model with repected to best metric.
./train_restore.sh ../configs/categorizing/hlstm_8_p_semb_ru_elmo_pre1024_focal_rur_add_hs512_f1.sh _F1_macro
```

## Evalution

```bash
# For evaluating from a trained model, sub_model_prefix follows the same guide as train_restore.sh
./dev.sh <config_file> sub_model_prefix

# dev with the saved model on dev test with respect to macro F1.
./dev.sh ../configs/categorizing/selected/C_C.sh _F1_macro

# dev on test means do the same evalution on test set.
./dev_on_test.sh ../configs/categorizing/C_C.sh _F1_macro
```

This scripts can be manually evoked once the model to be restored is saved
 in the "folder". After evaluation, a dev_{model_name}.log will
 generated in $CONFIG_DIR/training folder, and results on dev set will
 show in $CONFIG_DIR/results, results on test will show in
 $CONFIG_DIR/results_on_test

# Part II. Experiment Desgining
*******************************

The two tasks in our paper is distinguished by the following
configurations in the config file

All selected receipts are in `Expt/psyc-scripts/configs/categorizing/selected/`
and `Expt/psyc-scripts/configs/forecasting/selected/`.

You can follow the steps above to cook each of them.  Worth to
mention, if $VOCAB_DIR is already built, then please skip
preprocessing and preparing steps, only training and evalution are
required. If you would like to try diffrent tokenization or embedding,
then redo from the corresponding steps.

```bash
# categorization task will use the last utterance(response) to be labeled
# forecasting task will not use the last utterance(response) to be labeled
# `x` just means switch on, leave it empty for swith off
USE_RESPONSE_U=x

# We always use the speaker infomation for both context and response
USE_RESPONSE_S=x

# decode_goal in ['SPEAKER','ALL_LABEL','P_LABEL','T_LABEL','SEQ_TAG']
# use T_LABEL for therapist code only
DECODE_GOAL=T_LABEL
# use P_LABEL for patient code only
DECODE_GOAL=P_LABEL
```

We offer the performance table on the selected models in our paper as
follows. For more, description for each configuration, please refer to
[README for config file](Expt/psyc-scripts/configs/README.md)


For the name of selected models, last chaceracter 'C' or 'T' means client or therapist.
The second last character 'C' or 'F' means categorizing task or forecasting task.
The remaining part of the name is a id for distinguish differrent nerual architecture.
See more details in the paper

## Categorizing
For client, the best model does not need any word or utterance attention.

| Method                                                                                | macro    | FN           | CHANGE   | SUSTAIN  |
|---------------------------------------------------------------------------------------|:--------:|:------------:|:--------:|:--------:|
| Majority                                                                              | 30.6     | **__91.7__** | 0.0      | 0.0      |
| [Xiao et al. (2016)](http://scuba.usc.edu/pdf/xiao2016_behavioral-codi.pdf)           | 50.0     | 87.9         | 32.8     | __29.3__ |
| [BiGRU_generic_C](Expt/psyc_scripts/configs/categorizing/selected/BiGRU_generic_C.sh) | __50.2__ | 87.0         | __35.2__ | 28.4     |
| [BiGRU_ELMo_C](Expt/psyc_scripts/configs/categorizing/selected/BiGRU_ELMo_C.sh)       | 52.9     | 87.6         | **39.2** | 32.0     |
| [Can et al. (2015)](https://sail.usc.edu/publications/files/dogan-is150788.pdf)       | 44.0     | 91.0         | 20.0     | 21.0     |
| [Tanana et al. (2016)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4842096/)         | 48.3     | 89.0         | 29.0     | 27.0     |
| [CONCAT_C_C](Expt/psyc_scripts/configs/categorizing/selected/CONCAT_C_C.sh)           | 51.8     | 86.5         | 38.8     | 30.2     |
| [GMGRU_H_C_C](Expt/psyc_scripts/configs/categorizing/selected/GMGRU_H_C_C.sh)         | 52.6     | 89.5         | 37.1     | 31.1     |
| [BiDAF_H_C_C](Expt/psyc_scripts/configs/categorizing/selected/BiDAF_H_C_C.sh)         | 50.4     | 87.6         | 36.5     | 27.1     |
| [Our Best](Expt/psyc_scripts/configs/categorizing/selected/C_C.sh)                    | **53.9** | 89.6         | 39.1     | **33.1** |
| Change                                                                               | **+3.5** | **-2.1**     | **+3.9** | **+3.8** |


For the therapist, it uses GMGRUH for word attention and ANCHOR42 for utterance attention.

| Method                                                                                | macro    | FA       | RES      | REC      | GI       | QUC      | QUO      | MIA      | MIN       |
|---------------------------------------------------------------------------------------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| Majority                                                                              | 5.87     | 47.0     | 0.0      | 0.0      | 0.0      | 0.0      | 0.0      | 0.0      | 0.0       |
| [Xiao et al. (2016)](http://scuba.usc.edu/pdf/xiao2016_behavioral-codi.pdf)           | 59.3     | __94.7__ | 50.2     | 48.3     | 71.9     | 68.7     | 80.1     | 54.0     | 6.5       |
| [BiGRU_generic_T](Expt/psyc_scripts/configs/categorizing/selected/BiGRU_generic_T.sh) | __60.2__ | 94.5     | __50.5__ | __49.3__ | 72.0     | 70.7     | 80.1     | __54.0__ | __10.8__  |
| [BiGRU_ELMo_T](Expt/psyc_scripts/configs/categorizing/selected/BiGRU_ELMo_T.sh)       | 62.6     | 94.5     | 51.6     | 49.4     | 70.7     | 72.1     | 80.8     | 57.2     | 24.2      |
| [Can et al. (2015)](https://sail.usc.edu/publications/files/dogan-is150788.pdf)       | -        | 94.0     | 49.0     | 45.0     | __74.0__ | __72.0__ | __81.0__ | -        | -         |
| [Tanana et al. (2016)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4842096/)         | -        | 94.0     | 48.0     | 39.0     | 69.0     | 68.0     | 77.0     | -        | -         |
| [CONCAT_C_T](Expt/psyc_scripts/configs/categorizing/selected/CONCAT_C_T.sh)           | 61.0     | 94.5     | 54.6     | 34.3     | 73.3     | 73.6     | 81.4     | 54.6     | 22.0      |
| [GMGRU_H_C_T](Expt/psyc_scripts/configs/categorizing/selected/GMGRU_H_C_T.sh)         | 64.9     | 94.9     | **56.0** | 54.4     | **75.5** | **75.7** | **83.0** | **58.2** | 21.8      |
| [BiDAF_H_C_T](Expt/psyc_scripts/configs/categorizing/selected/BiDAF_H_C_T.sh)         | 63.8     | 94.7     | 55.9     | 49.7     | 75.4     | 73.8     | 80.0     | 56.2     | 24.0      |
| [Our Best](Expt/psyc_scripts/configs/categorizing/selected/C_T.sh)                    | **65.4** | **95.0** | 55.7     | **54.9** | 74.2     | 74.8     | 82.6     | 56.6     | **29.7**  |
| Change                                                                                | **+5.2** | **+0.3** | **+3.9** | **+3.8** | **+0.2** | **+2.8** | **+1.6** | **+2.6** | **+18.9** |


## Forecasting

For both client and therapist, the best model uses no word attention, and uses SELF42 utterance attention.

| Method                                                                       | Dev      | Dev      | Test     | Test | Test     | Test     |
|------------------------------------------------------------------------------|:--------:|:--------:|:--------:|:----:|:--------:|:--------:|
|                                                                              | CHANGE   | SUSTAIN  | macro    | FN   | CHANGE   | SUSTAIN  |
| [CONCAT_F_C](Expt/psyc_scripts/configs/forecasting/selected/CONCAT_F_C.sh)   | 20.4     | 30.2     | 43.6     | 84.4 | 23.0     | **23.5** |
| [HGRU_F_C](Expt/psyc_scripts/configs/forecasting/selected/HGRU_F_C.sh)       | 19.9     | 31.2     | **44.4** | 85.7 | **24.9** | 22.5     |
| [GMGRU_H_F_C](Expt/psyc_scripts/configs/forecasting/selected/GMGRU_H_F_C.sh) | 19.4     | 30.5     | 44.3     | 87.1 | 23.3     | 22.4     |
| [Forecast_C](Expt/psyc_scripts/configs/forecasting/selected/F_C.sh)          | **21.1** | **31.3** | 44.3     | 85.2 | 24.7     | 22.7     |


Except for R@3, all others are F1 score.

| Method                                                                                 | R@3      | macro    | FA       | RES      | REC      | GI       | QUC      | QUO      | MIA      | MIN      |
|:--------------------------------------------------------------------------------------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| [CONCAT_F_T](Expt/psyc_scripts/configs/forecasting/selected/CONCAT_F_T.sh)             | 72.5     | 23.5     | 63.5     | 0.6      | 0.0      | 53.7     | 27.0     | 15.0     | 18.2     | 9.0      |
| [HGRU_generic_F_T](Expt/psyc_scripts/configs/forecasting/selected/HGRU_generic_F_T.sh) | 76.8     | 24.0     | 71.0     | 2.7      | 20.5     | 58.8     | 27.5     | 12.9     | 15.2     | 1.6      |
| [HGRU_F_T](Expt/psyc_scripts/configs/forecasting/selected/HGRU_F_T.sh)                 | 76.0     | 28.6     | 71.4     | 12.7     | **24.9** | 58.3     | 28.8     | 5.9      | **17.4** | 9.7      |
| [GMGRU_H_F_T](Expt/psyc_scripts/configs/forecasting/selected/GMGRU_H_F_T.sh)           | 76.6     | 26.6     | **72.6** | 10.2     | 20.6     | 58.8     | 27.4     | 6.0      | 8.9      | 7.9      |
| [Forecase_T](Expt/psyc_scripts/configs/forecasting/selected/F_T.sh)                    | **77.0** | **31.1** | 71.9     | **19.5** | 24.7     | **59.2** | **29.1** | **16.4** | 15.2     | **12.8** |


# Part VI. Usage for Other Dataset or Tasks

## Building Data Input

   Preprocessing your own dataset into DSTC-like conversational json
   format is the main job to do before modeling.

   ```json
   [
    {
        "correct_seq_labels": [],
        "options-for-correct-answers": [
            {
                "tokenized_utterance": "it 's just",
                "codes": [
                    {
                        "origin_code": "GI",
                        "translated_code": "giving_info",
                        "coder_order": [
                            {
                                "order_id": 1,
                                "coder_id": "ms",
                                "cid": 72427
                            }
                        ]
                    }
                ],
                "uid": "(BAER_936)_31_5_T_49_51",
                "agg_label": "giving_info",
                "speaker": "T",
                "snt_id": 9878
            }
        ],
        "example-id": "(BAER_936)_(T, 27, 3)-(T, 31, 51)",
        "messages-so-far": [
            {
                "tokenized_utterance": "mm - hmm",
                "codes": [
                    {
                        "origin_code": "FA",
                        "translated_code": "facilitate",
                        "coder_order": [
                            {
                                "order_id": 1,
                                "coder_id": "ms",
                                "cid": 72411
                            }
                        ]
                    }
                ],
                "uid": "(BAER_936)_27_9_T_3_4",
                "agg_label": "facilitate",
                "speaker": "T",
                "snt_id": 5
            },
            ...
         ],
        "correct_labels": [
            3
        ],
        "pred_probs": [
            {
                "label_index": 2,
                "label_name": "reflection_complex",
                "prob": 0.2700542211532593
            },
            {
                "label_index": 3,
                "label_name": "reflection_simple",
                "prob": 0.100542211532593
            },
            ...
         ]
       },
       ...
   ]
   ```

   Our current code base is based on feeddict-based tensorflow inputs.
   In future, we will upgrade it with newer tensforflow feattures,
   such as estimator and tensorflow serving.

## Model Designing

   Our code base allows user to build converstational baseline models
   without writing much tensorflow code. For all supported model
   components, creating customized config file is the only thing to do
   for building a model for your dataset.

### Hierarchical Encoder

### Various Attention Mechansims

### Various Embeddings

   - Domain Specific Glove

   - Domain Specific ELMo

# Known Issues (To be moved to issues)

 1. Known issues about spaCy with python 2.7.5

  see https://github.com/explosion/spaCy/issues/3734, Please use python 2.7.12. But Python 2 will be dropped in Jan 2020, we will try to test our code on python 3 and publish a new repo for python 3.
