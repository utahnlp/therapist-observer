# Time-stamp: <>
# --------------------------------------------------------------------
# File Name          : match_layer.py
# Original Author    : jiessie.cao@gmail.com
# Description        :
# This module implements the core layer of Match-LSTM and BiDAF
# Extended on https://github.com/baidu/DuReader/blob/master/tensorflow/layers/match_layer.py
# Support multiple head and other attention mechanism
# --------------------------------------------------------------------

import tensorflow as tf
import tensorflow.contrib as tc

INF = 1e30

# Wilson: apply mask before real softmax
def softmax_mask(val, mask):
    return -INF * (1 - tf.cast(mask, tf.float32)) + val
# END

class MatchLSTMAttnCell(tc.rnn.LSTMCell):
    """
    Implements the Match-LSTM attention cell
    """
    def __init__(self, num_units, context_to_attend, mask, use_gate=False):
        super(MatchLSTMAttnCell, self).__init__(num_units, state_is_tuple=True)
        self.context_to_attend = context_to_attend
        self.fc_context = tc.layers.fully_connected(self.context_to_attend,
                                                    num_outputs=self._num_units,
                                                    activation_fn=None)
        self.use_gate = use_gate
        self.mask = mask

    def __call__(self, inputs, state, scope=None):
        (c_prev, h_prev) = state
        with tf.variable_scope(scope or type(self).__name__):
            ref_vector = tf.concat([inputs, h_prev], -1)
            G = tf.tanh(self.fc_context
                        + tf.expand_dims(tc.layers.fully_connected(ref_vector,
                                                                   num_outputs=self._num_units,
                                                                   activation_fn=None), 1))
            logits = tc.layers.fully_connected(G, num_outputs=1, activation_fn=None)
            # Wilson: TODO: Need to use mask
            # scores = tf.nn.softmax(logits, 1)
            logits = tf.reshape(logits, [-1, tf.shape(logits)[1]])
            scores = tf.nn.softmax(softmax_mask(logits, self.mask), 1)
            scores = tf.expand_dims(scores, 2)
            attended_context = tf.reduce_sum(self.context_to_attend * scores, axis=1)

# Wilson: apply a feed-forward network with sigmoid activation over the inputs and attended_context
# Like in R-Net
            if self.use_gate:
                new_inputs0 = tf.concat([inputs, attended_context], -1)
                dim = new_inputs0.get_shape().as_list()[-1]
                g = tc.layers.fully_connected(new_inputs0, num_outputs=dim, activation_fn=tf.nn.sigmoid)
                gnew_inputs0 = g * new_inputs0
                ginputs, gattended_context = tf.split(-1, 2, gnew_inputs0)
                new_inputs = tf.concat([ginputs, gattended_context,
                                        ginputs - gattended_context, ginputs * gattended_context],
                                       -1)
# END
            else:
                new_inputs = tf.concat([inputs, attended_context,
                                        inputs - attended_context, inputs * attended_context],
                                       -1)
            return super(MatchLSTMAttnCell, self).__call__(new_inputs, state, scope)

# Wilson: apply a feed-forward network with sigmoid activation over the inputs and attended_context, like in R-Net
class GatedMatchLSTMAttnCell(tc.rnn.LSTMCell):
    """
    Implements the Match-LSTM attention cell with gates
    """
    def __init__(self, num_units, context_to_attend, q_mask, use_gate=True):
        super(GatedMatchLSTMAttnCell, self).__init__(num_units, state_is_tuple=True)
        self.context_to_attend = context_to_attend
        self.fc_context = tc.layers.fully_connected(self.context_to_attend,
                                                    num_outputs=self._num_units,
                                                    activation_fn=None)
        self.use_gate = use_gate
        self.mask = q_mask

    def __call__(self, inputs, state, scope=None):
        (c_prev, h_prev) = state
        with tf.variable_scope(scope or type(self).__name__):
            ref_vector = tf.concat([inputs, h_prev], -1)
            G = tf.tanh(self.fc_context
                        + tf.expand_dims(tc.layers.fully_connected(ref_vector,
                                                                   num_outputs=self._num_units,
                                                                   activation_fn=None), 1))
            logits = tc.layers.fully_connected(G, num_outputs=1, activation_fn=None)
            # Wilson: TODO: Need to use q_mask
            # scores = tf.nn.softmax(logits, 1)
            logits = tf.reshape(logits, [-1, tf.shape(logits)[1]])
            scores = tf.nn.softmax(softmax_mask(logits, self.mask), 1)
            scores = tf.expand_dims(scores, 2)
            attended_context = tf.reduce_sum(self.context_to_attend * scores, axis=1)

            if self.use_gate:
                # dim = tf.shape(inputs)[-1]
                new_inputs0 = tf.concat([inputs, attended_context], -1)
                dim = new_inputs0.get_shape().as_list()[-1]
                g = tc.layers.fully_connected(new_inputs0, num_outputs=dim, activation_fn=tf.nn.sigmoid)
                new_inputs = g * new_inputs0
            else:
                new_inputs = tf.concat([inputs, attended_context], -1)
            return super(GatedMatchLSTMAttnCell, self).__call__(new_inputs, state, scope)


class GatedMatchLSTMLayer(object):
    """
    Implements the Match-LSTM layer, which attend to the question dynamically in a LSTM fashion.
    """
    def __init__(self, hidden_size, use_gate=True):
        self.hidden_size = hidden_size
        self.use_gate = use_gate

    def match(self, passage_encodes, question_encodes, p_length, q_length, p_mask, q_mask):
        """
        Match the passage_encodes with question_encodes using Match-LSTM algorithm
        """
        with tf.variable_scope('gated_match_lstm'):
            cell_fw = GatedMatchLSTMAttnCell(self.hidden_size, question_encodes, q_mask, self.use_gate)
            cell_bw = GatedMatchLSTMAttnCell(self.hidden_size, question_encodes, q_mask, self.use_gate)
            outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                             inputs=passage_encodes,
                                                             sequence_length=p_length,
                                                             dtype=tf.float32)
            match_outputs = tf.concat(outputs, 2)
            state_fw, state_bw = state
            c_fw, h_fw = state_fw
            c_bw, h_bw = state_bw
            match_state = tf.concat([h_fw, h_bw], 1)
        return match_outputs, match_state
# END

class MatchLSTMLayer(object):
    """
    Implements the Match-LSTM layer, which attend to the question dynamically in a LSTM fashion.
    """
    def __init__(self, hidden_size, use_gate=False):
        self.hidden_size = hidden_size
        self.use_gate = use_gate

    def match(self, passage_encodes, question_encodes, p_length, q_length, p_mask, q_mask):
        """
        Match the passage_encodes with question_encodes using Match-LSTM algorithm
        """
        with tf.variable_scope('match_lstm'):
            cell_fw = MatchLSTMAttnCell(self.hidden_size, question_encodes, q_mask, self.use_gate)
            cell_bw = MatchLSTMAttnCell(self.hidden_size, question_encodes, q_mask, self.use_gate)
            outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                             inputs=passage_encodes,
                                                             sequence_length=p_length,
                                                             dtype=tf.float32)
            match_outputs = tf.concat(outputs, 2)
            state_fw, state_bw = state
            c_fw, h_fw = state_fw
            c_bw, h_bw = state_bw
            match_state = tf.concat([h_fw, h_bw], 1)
        return match_outputs, match_state


class AttentionFlowMatchLayer(object):
    """
    Implements the Attention Flow layer,
    which computes Context-to-question Attention and question-to-context Attention
    """
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def match(self, passage_encodes, question_encodes, p_length, q_length, p_mask, q_mask):
        """
        Match the passage_encodes with question_encodes using Attention Flow Match algorithm
        """
##        # Wilson: softmax here needs to add mask?
##        JX = tf.shape(passage_encodes)[1]
##        with tf.variable_scope('bidaf'):
##            sim_matrix = tf.matmul(passage_encodes, question_encodes, transpose_b=True)
### Wilson: TODO: use p_mask and q_mask
##            #context2question_attn = tf.matmul(tf.nn.softmax(sim_matrix, -1), question_encodes)
###            c2q_scores = tf.multiply(tf.nn.softmax(sim_matrix, -1), tf.tile(tf.expand_dims(q_mask, axis=1), [1, tf.shape(passage_encodes)[1], 1]))
###            c2q_scores = c2q_scores/tf.reduce_sum(c2q_scores, axis=2, keep_dims=True)
###            context2question_attn = tf.matmul(c2q_scores, question_encodes)
##            # Expand q_mask into required tensor
##            mask = tf.tile(tf.expand_dims(q_mask, axis=1), [1, JX, 1])
##            weights = tf.nn.softmax(softmax_mask(sim_matrix, mask), -1)
##            context2question_attn = tf.matmul(weights, question_encodes)
##
### Wilson: TODO: use p_mask and q_mask
##            #b = tf.nn.softmax(tf.expand_dims(tf.reduce_max(sim_matrix, 2), 1), -1)
##            # last dimension should be over passage positions
##            b_logits = tf.expand_dims(tf.reduce_max(sim_matrix, 2), 1)
##            b = tf.nn.softmax(softmax_mask(b_logits, tf.expand_dims(p_mask, axis=1)), -1)
##
###            b = tf.multiply(b, tf.expand_dims(p_mask, axis=1))
###            b = b/tf.reduce_sum(b, axis=2, keep_dims=True)
        with tf.variable_scope('bidaf'):
            sim_matrix = tf.matmul(passage_encodes, question_encodes, transpose_b=True)
            context2question_attn = tf.matmul(tf.nn.softmax(sim_matrix, -1), question_encodes)
            b = tf.nn.softmax(tf.expand_dims(tf.reduce_max(sim_matrix, 2), 1), -1)
            question2context_attn = tf.tile(tf.matmul(b, passage_encodes),
                                         [1, tf.shape(passage_encodes)[1], 1])
            concat_outputs = tf.concat([passage_encodes, context2question_attn,
                                        passage_encodes * context2question_attn,
                                        passage_encodes * question2context_attn], -1)
            return concat_outputs, None

class GatedAttentionLayer(object):
    """
    Implements Gated Attention Layer as in GAReader
    """
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def match(self, passage_encodes, question_encodes, p_length, q_length, p_mask, q_mask):
        """
        Match the passage_encodes with question_encodes using Attention Flow Match algorithm
        """
        JX = tf.shape(passage_encodes)[1]
        with tf.variable_scope('gareader'):
            sim_matrix = tf.matmul(passage_encodes, question_encodes, transpose_b=True)
            mask = tf.tile(tf.expand_dims(q_mask, axis=1), [1, JX, 1])
            weights = tf.nn.softmax(softmax_mask(sim_matrix, mask), -1)

            # obtain query-aware passage encodings
            new_passage_encodes = tf.matmul(weights, question_encodes)

            return tf.multiply(passage_encodes, new_passage_encodes), None

class CollaborativeGatedMatchLSTMLayer(object):
    """
    Implements Collaborative GMLSTM layer
    """
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def match(self, passage_encodes, question_encodes, p_length, q_length, p_mask, q_mask):
        """
        Match the passage_encodes with question_encodes using Attention Flow Match algorithm
        """
        ga_layer = GatedMatchLSTMLayer(self.hidden_size)
        ga_layer1 = GatedMatchLSTMLayer(self.hidden_size)

        with tf.variable_scope('cgmlstm0'):
            # obtain passage-aware question encodes
            ga_question_encodes, _ =  ga_layer.match(question_encodes, passage_encodes, q_length, p_length, q_mask, p_mask)

        with tf.variable_scope('cgmlstm1'):
            # obtain question-aware passage encodes
            ga_passage_encodes, _ =  ga_layer1.match(passage_encodes, ga_question_encodes, p_length, q_length, p_mask, q_mask)

        return ga_passage_encodes, None

class CollaborativeGatedAttentionLayer(object):
    """
    Implements Collaborative Gated Attention Layer
    """
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def match(self, passage_encodes, question_encodes, p_length, q_length, p_mask, q_mask):
        """
        Match the passage_encodes with question_encodes using Attention Flow Match algorithm
        """
        ga_layer = GatedAttentionLayer(self.hidden_size)
        ga_layer1 = GatedAttentionLayer(self.hidden_size)

        with tf.variable_scope('cgareader0'):
            # obtain passage-aware question encodes
            ga_question_encodes, _ =  ga_layer.match(question_encodes, passage_encodes, q_length, p_length, q_mask, p_mask)

        with tf.variable_scope('cgareader1'):
            # obtain question-aware passage encodes
            ga_passage_encodes, _ =  ga_layer1.match(passage_encodes, ga_question_encodes, p_length, q_length, p_mask, q_mask)

        return ga_passage_encodes, None

# Not used
class SelfAttentionLayer(object):
    """
    Implements dot-product Attention layer over itself
    """
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def match(self, passage_encodes, p_mask):
        """
        Match the passage_encodes against itself
        """
        # get passage length
        # Shape of passage_encodes: [Batch Size * Num passages, Passage Length, Embedding Size]
        # Shape of p_mask:          [Batch Size * Num passages, Passage Length]
        JX = tf.shape(passage_encodes)[1]

        with tf.variable_scope('selfatt'):
            sim_matrix = tf.matmul(passage_encodes, passage_encodes, transpose_b=True)

            # Wilson: softmax here needs a mask
            # Expand p_mask into required tensor
            mask = tf.tile(tf.expand_dims(p_mask, axis=1), [1, JX, 1])
            logits = tf.nn.softmax(softmax_mask(sim_matrix, mask), -1)
            self_passage_encodes = tf.matmul(logits, passage_encodes)

            concat_outputs = tf.concat([passage_encodes, self_passage_encodes], -1)
            return concat_outputs, None

def dot_product_attention(Q, K, V, mask=None, dropout_keep_prob = 1.0):
    """
    Args:
    Q (tf.tensor): of shape (batch, q_size, d_model)
    K (tf.tensor): of shape (batch, k_size, d_model)
    V (tf.tensor): of shape (batch, k_size, d_model)
    mask (tf.tensor): of shape (batch, q_size, k_size)
    """
    assert Q.shape[-1] == K.shape[-1] == V.shape[-1]

    out = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # [batch, q_size, k_size]

    if mask is not None:
        # masking out (0.0) => setting to -inf.
        out = tf.multiply(out, mask) + (1.0 - mask) * (-1e10)

    out = tf.nn.softmax(out)  # [h * batch, q_size, k_size]
    out = tf.nn.dropout(out, dropout_keep_prob)
    out = tf.matmul(out, V)  # [h * batch, q_size, d_model]
    return out

def scaled_dot_product_attention(Q, K, V, dim_total, num_heads, mask=None, dropout_keep_prob = 1.0):
    """
    Args:
    Q (tf.tensor): of shape (num_heads * batch, q_size, d_model)
    K (tf.tensor): of shape (num_heads * batch, k_size, d_model)
    V (tf.tensor): of shape (num_heads * batch, k_size, d_model)
    mask (tf.tensor): of shape (num_heads * batch, q_size, k_size)
    """
    d = dim_total // num_heads
    assert d == Q.shape[-1] == K.shape[-1] == V.shape[-1]

    out = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # [h*batch, q_size, k_size]
    out = out / tf.sqrt(tf.cast(d, tf.float32))  # scaled by sqrt(d_k)

    if mask is not None:
        # masking out (0.0) => setting to -inf.
        out = tf.multiply(out, mask) + (1.0 - mask) * (-1e10)

    out = tf.nn.softmax(out)  # [num_heads * batch, q_size, k_size]
    out = tf.nn.dropout(out, dropout_keep_prob)
    out = tf.matmul(out, V)  # [num_heads * batch, q_size, d_model]
    return out

def multihead_attention(dim_total, num_heads, query, memory=None, mask=None, scope='attn', dropout_keep_prob = 1.0):
    """
    Args:
        dim_total : dim of total dimensions.
        num_heads : number of heads
        query (tf.tensor): of shape (batch, q_size, d_model)
        memory (tf.tensor): of shape (batch, m_size, d_model)
        mask (tf.tensor): shape (batch, q_size, k_size)
    Returns:h
        a tensor of shape (bs, q_size, d_model)
    """
    if memory is None:
            memory = query

    with tf.variable_scope(scope):
        # Linear project to d_model dimension: [batch, q_size/k_size, d_model]
        Q = tf.contrib.layers.fully_connected(query, dim_total, activation_fn=tf.nn.relu)
        K = tf.contrib.layers.fully_connected(memory, dim_total, activation_fn=tf.nn.relu)
        V = tf.contrib.layers.fully_connected(memory, dim_total, activation_fn=tf.nn.relu)

        # Split the matrix to multiple heads and then concatenate to have a larger
        # batch size: [h*batch, q_size/k_size, d_model/num_heads]
        Q_split = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
        K_split = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
        V_split = tf.concat(tf.split(V, num_heads, axis=2), axis=0)
        mask_split = tf.tile(mask, [num_heads, 1, 1])

        # Apply scaled dot product attention
        out = scaled_dot_product_attention(Q_split, K_split, V_split, dim_total, num_heads, mask_split, dropout_keep_prob)

        # Merge the multi-head back to the original shape
        out = tf.concat(tf.split(out, num_heads, axis=0), axis=2)  # [bs, q_size, d_model]

    return out
