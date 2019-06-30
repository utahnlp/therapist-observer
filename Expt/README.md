# Experiment Workspace: `Expt`

   `Expt` is the workspace for our experiments. Initially, it contains the following 4 main folders,
    We breifly give a overview of each of them, and the subfolders in them.

## `psyc-scripts`

    It contains all the bash scripts that organize the running of our
   experiments.  Worth to menthion, our principle of running neural
   experiments is seperating the reciept from its cook tool.  Hence,
   all cook tools are placed in `psyc-scripts/commands/`, and all cook
   receipts are placed in `psyc-scripts/configs`.

   During developing, build the core of cook tools in `tensorflow`
   code directory, and then wrap that as a configurable bash script in
   `psyc-scrpts/commands`. Then, write various receipts by setting
   different switch or hyperparamters. Finally, a cook tool(command)
   feeded by a receipt(config) will build a model as you wish.

    Worth to mention, a `env.sh` in commands folder is a special
   initialization code, which will set the important global variables
   that will used in our model.  It will be fine to run the code
   without any customization on env.sh However, please check the
   details in Expt/commands/env.sh script, which contains the global
   variables in our model.  Especially, please understand the usage
   for the ROOT_DIR, META_DATA_CONFIG, and RO_DATA_DIR, DATA_DIR.

##  `stats_scripts`

   Some pythong scripts are useful to monitor and make statistics for
   the training log, which can find the current best performance over
   each label, each metric, and each aggregation methion.

## `data`

   - `psyc_ro`

    It is a folder for read-only data, once data generated in this
   folder, it will keep unchanged, e.g. glove pretrained embedding,
   original data set, data splits etc.

       `meta_data_config` in it is about some meta configs for our
   data, due to data privacy, we only can publish the session ids used
   to generate our train/dev/test splits. This folder will configure
   in `Expt/psyc-scripts/commands/env.sh`, and will be used by
   trans.sh for data transformation.  Please use those ids for dataset
   splitting, so that the performance can be directly comparable.

   - `psyc`

   It the proprocessed data for traning, including tokenization, prepared training data.

## `workdir`

   It is folder to store all the experiment results organized by per folder one modeling receipt.

   - training.log or training_restore.log

   generated from `./train.sh` or `./train_restore.sh`, it shows all the training logs.
   By using `stats_scripts` above, you can analyze the log in realtime.
  ```bash
  # for analyzing training log for Patient(client) models
  python $ROOT_DIR/Expt/stats_scripts/stats_P.py train.log

  # for analyzing training log for Therapist models during resumed training.
  python $ROOT_DIR/Expt/stats_scripts/stats_T.py traini_restore.log
  ```

   - **models**

    models will save all the best models with repsect to each metrics to evaluate.
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

   - **results** and **results_on_test**

     results folder will list all the genenerated output predictions file.
     What's more, confusion matrix and its matlab figures will also genenerated there.
     results_on_test will show the result on test set.

   - **summary**

   This folder is for writing event and all kinds summaries, that can be read by tensorboard. 
