#!/usr/bin/python3
#\file    logging1.py
#\brief   Test of logging.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.23, 2025
import logging
from kbhit2 import *
from time_str2 import TimeStr
import datetime, time, random

def Main():
  log_filename= f'''/tmp/log_{TimeStr('normal_ms')}.txt'''
  logger= logging.getLogger('pp_repeat')
  logger.setLevel(logging.INFO)

  if logger.hasHandlers():
    logger.handlers.clear()

  formatter= logging.Formatter('%(asctime)s %(levelname)s: %(message)s')

  file_handler= logging.FileHandler(log_filename, mode='w')
  file_handler.setFormatter(formatter)
  logger.addHandler(file_handler)

  console_handler= logging.StreamHandler()
  console_handler.setFormatter(formatter)
  logger.addHandler(console_handler)

  logger.info('=== Main() started ===')
  logger.info(f'Program name: {__file__}')
  logger.info(f'Log file: {log_filename}')
  logger.info(f'Start time: {datetime.datetime.now().isoformat()}')

  total_cycles= 0
  cycle_durations= []
  for cycle in range(1, 6):
    total_cycles+= 1
    t0= time.time()

    logger.info(f'Attempt #{cycle}')
    success= random.choice([True, False])

    if not success:
      logger.warning(f'Attempt #{cycle} failed, retrying')
      time.sleep(0.5)
      logger.info(f'Attempt #{cycle} retry succeeded')
    else:
      logger.info(f'Attempt #{cycle} succeeded')

    time.sleep(0.5)

    if cycle==3:
      logger.warning('Operator interruption - pausing operation')
      time.sleep(1)
      logger.info('Operation resumed by operator')

    cycle_duration= time.time()-t0
    logger.info(f'Cycle time (s): {cycle_duration}')
    cycle_durations.append(cycle_duration)

  logger.info('=== Main() ended ===')
  logger.info(f'File name: {__file__}')
  logger.info(f'Total cycles: {total_cycles}')
  avg_time= sum(cycle_durations)/len(cycle_durations)
  logger.info(f'Average cycle time (s): {avg_time:.2f}')

  file_handler.close()
  logger.removeHandler(file_handler)


if __name__=='__main__':
  with TKBHit() as kbhit:
    while True:
      if not kbhit.IsActive():
        break
      sys.stdout.write('q or space > ')
      sys.stdout.flush()
      key= kbhit.KBHit(timeout=10)
      sys.stdout.write('\n')
      if key=='q':
        break
      if key==' ':
        Main()

