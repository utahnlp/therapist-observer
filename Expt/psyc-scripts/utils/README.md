# Utils
This folder contains the tools that may be useful for mananger experiments, shuf and split large dataset, extracting snt list for glove training etc.

## Manage Experiments

When you finished developing an new idea of neural components or some
hyperparameters for tuning. We need feed the command those new
configurtions.  In our framework for experiment managing, commands are
the universal interface for running a job, while it can take various
model receipts. For every hyperparameters and every switch-off to
customize the nerual computing graph, there is a correspoding argument
in both commands and configs files. Every time you add a new argument
in your `tensorflow/classes/configure_reader.py`, then you want the
arguments can be set in the experiments, which needs you add this
arguments in both the configuration file and the commands files. As
you have so many configuration and commands file to be updated. That's
boring.
    `./add_args.sh` and `remove_args.sh` are designed to add or remove arguments to all the configuration files and all the command files to make the arguments take effect. It take 2 arguments, the first is the folder for "configs", the second it the folder for "commands"

  -**add_args.sh**
```
# STEP 1, edit your add_args.sh, for ARG_NAME, ARG_COMMENTS, lc_arg_name to be added,
# all UPPER case means this string is used in configuration file, while lowercase is used in the commands

cd utils

# STEP 2, add arguments into all configs and commands files. Especially, you can change any subdirectories for your configs by using subdir paths
./add_args.sh ../configs/ ../commands

# STEP 3, if the default value for added command should be changed in some specific configurations, then just update the specific one

```

  -**remove_args.sh**
```
# STEP 1, edit your remove_args.sh, for ARG_NAME, ARG_COMMENTS, lc_arg_name to be added, 
# all UPPER case means this string is used in configuration file, while lowercase is used in the commands
Pay attention to the patterns we used in sed. Feel free to extend those patterns if necessary. 

cd utils
./remove_args.sh ../configs/ ../commands
```

## Shuf and Split Large Dataset

   - **shuf_and_split.sh**

When training shuffle is required. However, a dataset is too large to
load into memory and then shuffing. This script try to split the
large dataset and make the shuf happens hierarchically. One inter
shuffing is between differernt splits, another inner shuffing is
within each split. Only for large split size, this method may mimic
the global shuffing for large dataset. In our experiment, it is used
when doing multiple tasks with other large dataset, while it is not
published in our paper. We leave it here for future usage.


## Sentence Extraction

   - **make_snt_list.sh**

This script will only be used when you to want to build domain
specific glove or ELMo embedding. It can be used to extract all
sentence list from the general psychotherapy corpus - Alexrander
Street Press. Please see to Glove for main about training custmized
embedding.
