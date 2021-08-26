#!/usr/bin/python
#\file    gen_mg1.py
#\brief   Generate data from MG data directory.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.26, 2021
import sys
import os
import re
import yaml
import cv2
import numpy as np

if __name__=='__main__':
  data_dir= sys.argv[1]
  label_suffix= '-gdw.dat'
  fv1_name= '-fv-fvo2_2-MV1'
  fv2_name= '-fv-fvo2_2r-MV1'
  label_key= 'weight'
  category_key= 'g_ins'
  tool_name= 'S3c'
  out_dir_fmt= 'data_generated/mg1/{tool_name}/{cat}/{train_test}/{il}'
  train_data_ratio= 0.7
  out_img_ext= '.png'

  data_files= os.listdir(data_dir)
  label_files= sorted(f for f in data_files if f.endswith(label_suffix))
  fv1_files= [next(f for f in data_files if f.startswith(lf[:-len(label_suffix)]+fv1_name))
                for lf in label_files]
  fv2_files= [next(f for f in data_files if f.startswith(lf[:-len(label_suffix)]+fv2_name))
                for lf in label_files]
  print 'Found {} data'.format(len(label_files))

  labels_raw= [yaml.load(open(os.path.join(data_dir,lf),'r')) for lf in label_files]
  categories= set(lr[category_key] for lr in labels_raw)
  cat_datalen= {cat:sum(lr[category_key]==cat for lr in labels_raw) for cat in categories}

  print 'Found {} categories'.format(len(categories))
  for cat in categories:
    print '  {}: {} data'.format(cat, cat_datalen[cat])

  for cat in categories:
    for train_test in ('train','test'):
      for il in ('images','labels'):
        try:
          os.makedirs(out_dir_fmt.format(tool_name=tool_name, cat=cat, train_test=train_test, il=il))
        except:
          pass
    n_data= 0
    for fv1,fv2,lf,lr in zip(fv1_files,fv2_files,label_files,labels_raw):
      if lr[category_key]!=cat:  continue
      train_test= 'train' if n_data<cat_datalen[cat]*train_data_ratio else 'test'
      out_dir= out_dir_fmt.format(tool_name=tool_name, cat=cat, train_test=train_test, il='{il}')
      img_filename= lf[:-len(label_suffix)]+out_img_ext
      label_filename= img_filename+'.dat'
      img_filepath= os.path.join(out_dir.format(il='images'),img_filename)
      label_filepath= os.path.join(out_dir.format(il='labels'),label_filename)
      #image: compose fv1,fv2
      #label: lr[label_key]
      img1= cv2.imread(os.path.join(data_dir,fv1))
      img2= cv2.imread(os.path.join(data_dir,fv2))
      image= np.concatenate((img1,img2), axis=1)
      cv2.imwrite(img_filepath, image)
      with open(label_filepath,'w') as fp:
        fp.write('{0}\n'.format(lr[label_key]))
      print 'Saved data: ', img_filepath, label_filepath
      n_data+= 1

