from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

pic_path = r'./NTHU.png'
augment_path = r'./NTHU_data_agument'

data_gen = ImageDataGenerator(
             rotation_range = 30,
             width_shift_range = 0.1,
             height_shift_range = 0.1,
             zoom_range=0.2,
             fill_mode='nearest'
            )

img = load_img(pic_path)
x = img_to_array(img)
x = x.reshape((1,)+x.shape)
n = 1
for batch in data_gen.flow(x, batch_size=1, save_to_dir=augment_path, save_prefix='train', save_format='png'):
   n+=1
   print(n)
   if n > 100:
      break    
