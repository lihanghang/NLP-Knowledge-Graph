import tensorflow as tf


'''
input
'''

batch_size = 5
encoder_input = tf.placeholder(shape=[batch_size, None], dtype=tf.int32)
decoder_target = tf.placeholder(shape=[batch_size, None], dtype=tf.int32)

'''
Embedding
'''

input_vocab_size = 10
target_vocab_size = 10
input_embedding_size = 20
target_embedding_size = 20


encoder_input = tf.one_hot(encoder_input, depth=input_vocab_size, dtype=tf.int32)
decoder_target = tf.one_hot(decoder_target, depth=target_vocab_size, dtype=tf.int32)

input_embedding = tf.Variable(tf.random_uniform(shape=[input_vocab_size, input_embedding_size]))
target_embedding = tf.Variable(tf.random_uniform(shape=[target_vocab_size, target_embedding_size], minval=-1.0), dtype=tf.float32)

input_embedd = tf.nn.embedding_lookup(input_embedding, encoder_input)
target_embedd = tf.nn.embedding_lookup(target_embedding, decoder_target)


'''
Encoder
'''

rnn_hidden_size = 20
cell = tf.nn.rnn_cell.GRUCell(num_units=rnn_hidden_size)
init_state = cell.zero_state(batch_size, dtype=tf.float32)
encoder_output, encoder_state = tf.nn.dynamic_rnn(cell, input_embedd, initial_state=init_state)


'''
Decoder
'''

inference = False
seq_len = tf.constant([3, 4, 5, 2, 3], tf.int32)
if not inference:
    helper = tf.contrib.seq2seq.TrainingHelper(target_embedd, sequence_length=seq_len)
else:
    helper = tf.contrib.seq2seq.InferenceHelper(target_embedd, sequence_length=seq_len)

d_cell = tf.nn.rnn_cell.GRUCell(rnn_hidden_size)

'''
attention cell
'''

attetion_mechanism = tf.contrib.seq2seq.BandanauAttention(rnn_hidden_size, encoder_output)
decoder_cell = tf.contrib.seq2seq.AttentionWrapper(d_cell, attetion_mechanism, attention_layer_size=rnn_hidden_size)
de_state = decoder_cell.zero_state(batch_size, dtype=tf.float32)
out_cell = tf.contrib.rnn.OutputPojectionWrapper(decoder_cell, target_vocab_size)

with tf.variable_scope("decoder"):
    decoder = tf.contrib.seq2seq.BasicDecoder(
        out_cell,
        helper,
        de_state,
        tf.layers.Dense(target_vocab_size)
    )


'''
dynamic decoder
'''

final_outputs, final_state, final_seq_len = tf.contrib.seq2seq.dynamic_decoder(decoder, swamp_memory=True)

