import os
from PIL import Image
import tensorflow as tf

cwd =r'./NCCU_serialize_image'
tfrecord_file_path = r'./'

#each tfrecord save image count
bestnum = 100
# the number of image
num = 0
# the number of tfrecord
recordfilenum = 0
# prepare for label
classes = []
for i in os.listdir(cwd):
    classes.append(i)
    
tfrecordfilename = ("traindata_NCCU.tfrecords-%.3d" % recordfilenum)
writer = tf.python_io.TFRecordWriter(os.path.join(tfrecord_file_path , tfrecordfilename))

for index, name in enumerate(classes):
    class_path = os.path.join(cwd, name)
    for img_name in os.listdir(class_path):
        num = num +1
        if num > bestnum:
            num = 1
            recordfilenum += 1
            tfrecordfilename = ("traindata_NCCU.tfrecords-%.3d"% recordfilenum)
            writer = tf.python_io.TFRecordWriter(os.path.join(tfrecord_file_path , tfrecordfilename))
        img_path = os.path.join(class_path, img_name)
        img = Image.open(img_path, 'r')
        img_raw = img.tobytes()
        example = tf.train.Example(
            features = tf.train.Features(
                feature={
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'img_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                    }
                )            
            )
        writer.write(example.SerializeToString())
writer.close()       
            

