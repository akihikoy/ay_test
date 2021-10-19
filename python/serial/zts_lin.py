#!/usr/bin/python
#\file    zts_lin.py
#\brief   Reading values simultaneously from an IMADA ZTS force sensor and a linear encoder on Arduino.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.19, 2021
from IMADA_ZTS2 import TZTS
from encoder1 import TLinearEncoder
import sys
import time

if __name__=='__main__':
  log_file_name= sys.argv[1] if len(sys.argv)>1 else None
  dev_zts= sys.argv[2] if len(sys.argv)>2 else '/dev/ttyS6'  #COM6(Win); /dev/ttyACM0
  dev_lin= sys.argv[3] if len(sys.argv)>3 else '/dev/ttyS5'  #COM5(Win); /dev/ttyACM0

  zts= TZTS(dev_zts, cnt=True)
  lin= TLinearEncoder(dev_lin)
  try:
    log_fp= open(log_file_name, 'w') if log_file_name else None
    while True:
      time.sleep(0.01)
      print '{n_zts:06d} {n_lin:06d} Latest: {v_zts:9.04f}{u_zts} {v_lin:9.04f}'.format(n_zts=zts.N, n_lin=lin.N, v_zts=zts.Value, u_zts=zts.Unit, v_lin=lin.Value)
      if log_fp:  log_fp.write('{} {} {} {}\n'.format(time.time(), zts.Value, zts.Unit, lin.Value))

  except KeyboardInterrupt:
    pass

  finally:
    zts.Stop()
    lin.Stop()
    if log_fp:
      log_fp.close()
      print 'Logged into:',log_file_name

