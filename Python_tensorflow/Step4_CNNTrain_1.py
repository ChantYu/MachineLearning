import tensorflow as tf
import tensorflow.contrib.slim.nets as nets


def read_and_decode_tfrecord(filename):
    filename_deque = tf.train.string_input_producer(filename)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_deque)
    features = tf.parse_single_example(serialized_example, features={
        'label': tf.FixedLenFeature([], tf.int64),
        'img_raw': tf.FixedLenFeature([], tf.string)})
    label = tf.cast(features['label'], tf.int32)
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [64, 64, 3])
    img = tf.cast(img, tf.float32) / 255.0       
    return img, label


save_dir = r"./train_image.model"
batch_size_ = 2
lr = tf.Variable(0.0001, dtype=tf.float32)
x = tf.placeholder(tf.float32, [None, 64, 64, 3])
y_ = tf.placeholder(tf.float32, [None])

#train_list = ['traindata_63.tfrecords-000', 'traindata_63.tfrecords-001', 'traindata_63.tfrecords-002',
#              'traindata_63.tfrecords-003', 'traindata_63.tfrecords-004', 'traindata_63.tfrecords-005',
#              'traindata_63.tfrecords-006', 'traindata_63.tfrecords-007', 'traindata_63.tfrecords-008',
#              'traindata_63.tfrecords-009', 'traindata_63.tfrecords-010', 'traindata_63.tfrecords-011',
#              'traindata_63.tfrecords-012', 'traindata_63.tfrecords-013', 'traindata_63.tfrecords-014',
#              'traindata_63.tfrecords-015', 'traindata_63.tfrecords-016', 'traindata_63.tfrecords-017',
#              'traindata_63.tfrecords-018', 'traindata_63.tfrecords-019', 'traindata_63.tfrecords-020',
#              'traindata_63.tfrecords-021']

train_list = ['NCTU_traindata.tfrecords-000','NTHU_traindata.tfrecords-000','NCCU_traindata.tfrecords-000','NTU_traindata.tfrecords-000']
# Radom series
img, label = read_and_decode_tfrecord(train_list)
img_batch, label_batch = tf.train.shuffle_batch([img, label], num_threads=2, batch_size=batch_size_, capacity=10000,
                                                min_after_dequeue=9900)

# Labeling - onehot
one_hot_labels = tf.one_hot(indices=tf.cast(y_, tf.int32), depth=100)
pred, end_points = nets.resnet_v2.resnet_v2_50(x, num_classes=100, is_training=True)
pred = tf.reshape(pred, shape=[-1, 100])

# Loss function , optimizer
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=one_hot_labels))
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

# accurate
a = tf.argmax(pred, 1)
b = tf.argmax(one_hot_labels, 1)
correct_pred = tf.equal(a, b)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    i = 0
    while True:
        i += 1
        b_image, b_label = sess.run([img_batch, label_batch])
        _, loss_, y_t, y_p, a_, b_ = sess.run([optimizer, loss, one_hot_labels, pred, a, b], feed_dict={x: b_image,
                                                                                                        y_: b_label})
        print('step: {}, train_loss: {}'.format(i, loss_))
        if i % 20 == 0:
            _loss, acc_train = sess.run([loss, accuracy], feed_dict={x: b_image, y_: b_label})
            print('--------------------------------------------------------')
            print('step: {}  train_acc: {}  loss: {}'.format(i, acc_train, _loss))
            print('--------------------------------------------------------')
            if i == 100:
                saver.save(sess, save_dir, global_step=i)
            elif i == 200:
                saver.save(sess, save_dir, global_step=i)
            elif i == 300:
                saver.save(sess, save_dir, global_step=i)
                break
    coord.request_stop()
    coord.join(threads)
