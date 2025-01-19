#!/usr/bin/python3
#\file    dict_shortstr.py
#\brief   Dictionary to yaml to short base64 string (encoding), and decoding.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jul.29, 2021
import yaml
import base64
import zlib

def EncodeDictB64(d, with_compress):
  d_yaml= yaml.dump(d)
  if with_compress:  d_yaml_c= zlib.compress(d_yaml.encode('utf-8'))
  else:  d_yaml_c= d_yaml.encode('utf-8')
  d_b64= base64.b64encode(d_yaml_c).decode('utf-8')
  return d_b64

def DecodeDictB64(d_b64, with_compress):
  d_yaml_c= base64.b64decode(d_b64.encode('utf-8'))
  if with_compress:  d_yaml= zlib.decompress(d_yaml_c)
  else:  d_yaml= d_yaml_c
  d= yaml.load(d_yaml.decode('utf-8'), yaml.SafeLoader)
  return d

if __name__=='__main__':
  def Test(d, name):
    print('{0}= {1}'.format(name,d))
    d_b64_wc= EncodeDictB64(d, True)
    d_b64_woc= EncodeDictB64(d, False)
    if DecodeDictB64(d_b64_wc, True)!=d:
      print('EncodeDictB64/with_compress failed.')
      print('d_b64_wc=',d_b64_wc)
      raise Exception()
    if DecodeDictB64(d_b64_woc, False)!=d:
      print('EncodeDictB64/without_compress failed.')
      print('d_b64_woc=',d_b64_woc)
      raise Exception()
    print('  d_b64(with_compress)=',d_b64_wc,'len=',len(d_b64_wc))
    print('  d_b64(without_compress)=',d_b64_woc,'len=',len(d_b64_woc))

  d1= {
    'a': 10,
    'b': -125,
    }
  Test(d1, 'd1')

  d2= {
    'Brightness': -6,
    'Contrast': 53,
    'Saturation': 50,
    'Hue': 0,
    'White Balance Temperature, Auto': 0,
    'Gamma': 45,
    'Gain': 0,
    'White Balance Temperature': 6300,
    'Sharpness': 20,
    'Backlight Compensation': 0,
    'Exposure, Auto': 1,
    'Exposure (Absolute)': 1000,
    'Iris, Absolute': 5,
    }
  Test(d2, 'd2')

