#!/bin/bash
#\file    sync_models.sh
#\brief   Synchronizing the learned model files.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.27, 2021

server=$1
rsync -av -e 'ssh' $server:'prg/ay_test/python/torch/model_learned' .

