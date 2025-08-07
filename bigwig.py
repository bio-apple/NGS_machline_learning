import tensorflow as tf

# DNA string: 'ACGTN'
def one_hot_encode(seq):
    mapping = tf.constant([[1,0,0,0,0],   # A
                           [0,1,0,0,0],   # C
                           [0,0,1,0,0],   # G
                           [0,0,0,1,0],   # T
                           [0,0,0,0,1]])  # N
    vocab = tf.constant(list("ACGTN"))
    table = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(vocab, tf.range(5)),
        default_value=4
    )
    seq_idx = table.lookup(tf.strings.bytes_split(seq))
    return tf.gather(mapping, seq_idx)

# 示例
one_hot = one_hot_encode("ATCGGGGGG")
print(one_hot.numpy())
