from PIL import Image
import os

img_path     = r'./NCCU_data_agument'
resize_path  = r'./NCCU_resize_image'

for i in os.listdir(img_path):
   im  = Image.open(os.path.join(img_path,i))
   out = im.resize((64,64))
   if not os.path.exists(resize_path):
      os.makedirs(resize_path)
   out.save(os.path.join(resize_path, i))
   