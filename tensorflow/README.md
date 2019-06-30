# Main Code Directory: tensorflow

  Main code dir, includes all the dataset loader, paraphrase loader, models
### dial_run.py

   The main entry and command dispather for our system. All above commands will dispacthed to different programs.

### vocab.py

   Vocabulary is used for a token2id mapping. It is used in preparing phase. 

### dial_dataset.py

   Data loader for dialogue dataset. It will handle dataset loading, mini-batches assemblying, token2id, and padding processing. It will generate mini-batchs for feed_dict. 

### paraphrase_dataset.py

   Data loader and tools for paraphrase look-up.  

### dial_model.py

   Core of the dialouge models.  

### math_utils.py

   Some math utils for handling math out of the tensorflow.

### layers
   All the RNN, BIDAF attention model utils for tensorflow.

## utils:

  mainly for some asisitant utils, include tokenization and paraphrase utils

# ELMo Usage (elmo_utils.py)
Our ELMo componenet seprate the downstream with the main model. All elmo-related opertations are placed into the 'elmo_utils.py'. Every time, you want to use elmo, just hold an elmo_utils instance, then all is done.


For simpliying ElMo Usage, we use three functions in elmo_utils.py to introduce the elmo, which contains the following three steps:
```
Step 1. def get_elmo_char_ids(self, sentences)

"sentences" are List[List[str]], basicly list of tokenized sentences.

Return : [sentence_num, token_num, max_char_num]

This tensor will be feed into the bilm graph (Step 2), assigned to the input_ids_place_holder


Step 2. def get_elmo_emb_op(self, input_ids_place_holder):

input_ids_place_holder : [sentence_num, token_num, max_char_num

Given the input ids place holder, reutrn a ops for computing the language model
        {
         'lm_embeddings': embedding_op, (Batch_Size, options['lstm']['n_layers'] + 1, None, 2 * options['lstm']['projection_dim'])
         'lengths': sequence_lengths_op, (Batch_Size, )
         'mask': op to compute mask (Batch_Size, max_sentence length)
        }

Step 3. def weight_layers(self, name, bilm_ops, l2_coef=None, use_top_only=False, do_layer_norm=False):
Ops in the step 2 will feed into the step 3. Step 3 will be trained with downstream task.
While task 1 and 2 is fixed and only forward inference. 
return
{
    'weighted_op': op to compute weighted average for output, [Batch_Size, max_length, output_dim]
    'regularization_op': op to compute regularization term
}

```

In our system, there are 120+ sentences to go forward ELMo networks, which is too low, and make the code hard to write and manage. We use 2 kinds of elmo cache for this dialogue system.

```
Cache 1. Utterance_Cache, <snt_id, emb>  (def build_elmo_char_cache) 
   In this cache, every entry is correpsonding to the elmo emebedding of a sentence.
   all candidate answers, passage utterance, questgion utterance are in this cache. 

Cache 2. Passage and Question Cache  <sample_id, emb>, (def build_elmo_cache_for_samples )

(These two depend on the context_window, and question_window, which not be a single sentence.)

Method 1, we still can prepapre passage and question as single sentences with context and qustion window. 
Method 2, add new cache, indexed by the id of sample. Then given a sample, we can find their contactnated P and concated Q. We use the Method 2 here. 

```

Both cache will be prepared in the prepare.sh, please see more details on `dial_run.py`, function prepare()

During the preparing, it requires going through all the sentences, all samples to prepare utterance and P, Q cache.

For all these 3 cache, they will follow the step 1 and 2 to store the ops in the cache, which can be directly used in he downstream task during the training and validation. Some details in elmo_utils.py

Hence, after prepare the cache, downstream task will need the 3 cache file to run.
