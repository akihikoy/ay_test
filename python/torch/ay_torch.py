#!/usr/bin/python
#\file    ay_torch.py
#\brief   PyTorch utility.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Nov.24, 2021
import numpy as np
import torch
import torchvision
import torchinfo
import time
import copy
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from PIL import Image as PILImage

#Container class to share variables.
class TContainer(object):
  def __init__(self):
    pass
  def __del__(self):
    pass
  def __str__(self):
    return str(self.__dict__)
  def __repr__(self):
    return str(self.__dict__)
  def __iter__(self):
    return self.__dict__.itervalues()
  def items(self):
    return self.__dict__.items()
  def iteritems(self):
    return self.__dict__.iteritems()
  def keys(self):
    return self.__dict__.keys()
  def values(self):
    return self.__dict__.values()
  def __getitem__(self,key):
    return self.__dict__[key]
  def __setitem__(self,key,value):
    self.__dict__[key]= value
  def __delitem__(self,key):
    del self.__dict__[key]
  def __contains__(self,key):
    return key in self.__dict__
  def Cleanup(self):
    keys= self.__dict__.keys()
    for k in keys:
      self.__dict__[k]= None
      del self.__dict__[k]

def MergeDict(d_base, d_new, allow_new_key=True):
  if isinstance(d_new, (list,tuple)):
    for d_new_i in d_new:
      MergeDict(d_base, d_new_i)
  else:
    for k_new,v_new in d_new.iteritems() if hasattr(d_new,'iteritems') else d_new.items():
      if not allow_new_key and k_new not in d_base:
        raise Exception('MergeDict: Unexpected key:',k_new)
      if k_new in d_base and (isinstance(v_new,dict) and isinstance(d_base[k_new],dict)):
        MergeDict(d_base[k_new], v_new)
      else:
        d_base[k_new]= v_new
  return d_base  #NOTE: d_base is overwritten. Returning it is for the convenience.

def MergeDictSum(d_base, d_new, allow_new_key=True):
  if isinstance(d_new, (list,tuple)):
    for d_new_i in d_new:
      MergeDictSum(d_base, d_new_i)
  else:
    for k_new,v_new in d_new.iteritems() if hasattr(d_new,'iteritems') else d_new.items():
      if not allow_new_key and k_new not in d_base:
        raise Exception('MergeDict: Unexpected key:',k_new)
      if k_new in d_base and (isinstance(v_new,dict) and isinstance(d_base[k_new],dict)):
        MergeDictSum(d_base[k_new], v_new)
      else:
        if k_new in d_base:  d_base[k_new]+= v_new
        else:  d_base[k_new]= v_new
  return d_base  #NOTE: d_base is overwritten. Returning it is for the convenience.

class TFuncList(list):
  def __init__(self, *args, **kwargs):
    super(TFuncList,self).__init__(*args, **kwargs)
  def __call__(self, *args, **kwargs):
    return [f(*args, **kwargs) for f in self]
  def __add__(self, rhs):
    if isinstance(rhs,(TFuncList,list)):
      return TFuncList(super(TFuncList,self).__add__(rhs))
    else:
      return TFuncList(super(TFuncList,self).__add__([rhs]))
  def __iadd__(self, rhs):
    if isinstance(rhs,(TFuncList,list)):
      return super(TFuncList,self).__iadd__(rhs)
    else:
      return super(TFuncList,self).__iadd__([rhs])

'''A callback interface class.'''
class TCallbacks(object):
  def Callbacks(self):
    return {fname[3:]:getattr(self,fname) for fname in dir(self) if fname.startswith('cb_')}

class TLogger(TCallbacks):
  def __init__(self):
    self.time_train= []
    self.time_test= []
    self.loss_train= []
    self.loss_test= []
    self.metric_train= []
    self.metric_test= []
    self.lr= []

  def cb_epoch_train_begin(self, l):
    self.t0= time.time()
  def cb_epoch_train_end(self, l):
    self.time_train.append(time.time()-self.t0)
    if l.loss is not None:  self.loss_train.append(l.loss)
    if l.metric is not None:  self.metric_train.append(l.metric)
  def cb_epoch_test_begin(self, l):
    self.t0= time.time()
  def cb_epoch_test_end(self, l):
    self.time_test.append(time.time()-self.t0)
    if l.loss:  self.loss_test.append(l.loss)
    if l.metric:  self.metric_test.append(l.metric)
  def cb_batch_train_end(self, l):
    self.lr.append([param_group['lr'] for param_group in l.opt.param_groups])

  def Plot(self):
    fig= plt.figure(figsize=(10,5))
    ax_loss= fig.add_subplot(1,2,1,title='Learning curve (loss)',xlabel='epoch',ylabel='loss')
    ax_loss.plot(range(len(self.loss_train)), self.loss_train, color='blue', label='loss(train)')
    ax_loss.plot(range(len(self.loss_test)), self.loss_test, color='red', label='loss(test)')
    ax_loss.set_yscale('log')
    ax_loss.legend()
    ax_metric= fig.add_subplot(1,2,2,title='Learning curve (metric)',xlabel='epoch',ylabel='metric')
    ax_metric.plot(range(len(self.metric_train)), self.metric_train, color='blue', label='metric(train)')
    ax_metric.plot(range(len(self.metric_test)), self.metric_test, color='red', label='metric(test)')
    ax_metric.set_yscale('log')
    ax_metric.legend()
    plt.show()

  def PlotLR(self):
    fig= plt.figure()
    ax_lr= fig.add_subplot(1,1,1,title='Learning rate',xlabel='iteration',ylabel='lr')
    ax_lr.plot(range(len(self.lr)), self.lr, color='blue', label='lr')
    ax_lr.legend()
    plt.show()

class TDisp(TCallbacks):
  def cb_fit_begin(self, l):
    print('loss(train)\tloss(test)\tmetric(test)\ttime')
  def cb_epoch_train_begin(self, l):
    self.t0= time.time()
  def cb_epoch_train_end(self, l):
    self.time_train= time.time()-self.t0
    self.loss_train= l.loss
  def cb_epoch_test_begin(self, l):
    self.t0= time.time()
  def cb_epoch_test_end(self, l):
    self.time_test= time.time()-self.t0
    self.loss_test= l.loss
    self.metric_test= l.metric
    print(f'{self.loss_train:.8f}\t{self.loss_test:.8f}\t{self.metric_test:.8f}\t{self.time_train+self.time_test:.6f}')

'''
Prediction helper.
'''
def PredBatch(net, batch, tfm_batch=None, device=None, with_x=True, with_y=True):
  if device is None:  device= torch.device('cpu')
  if next(net.parameters()).device != device:
    net.to(device)
  x,y= tfm_batch(batch)
  if isinstance(x,(tuple,list)):
    x= (xi.to(device) for xi in x)
    pred= net(*x)
  else:
    x= x.to(device)
    pred= net(x)
  if with_y:
    y= tuple(yi.to(device) for yi in y) if isinstance(y,(tuple,list)) else y.to(device)
  if with_x:
    if with_y:  return x,y,pred
    else:       return x,pred
  else:
    if with_y:  return y,pred
    else:       return pred

'''
Evaluation helper.
'''
def Eval(net, x, device=None):
  if device is None:  device= torch.device('cpu')
  if next(net.parameters()).device != device:
    net.to(device)
  net.eval()
  with torch.no_grad():
    if isinstance(x,tuple):
      x= (torch.stack(xi).to(device) if isinstance(xi,(tuple,list)) else xi.to(device) for xi in x)
      pred= net(*x)
    else:
      x= torch.stack(x).to(device) if isinstance(x,(tuple,list)) else x.to(device)
      pred= net(x)
  return pred

'''
Helper to assign parameters.
'''
def AssignParamGroups(obj, key, params):
  for i, param_group in enumerate(obj.param_groups):
    param_group[key]= params if isinstance(params,(int,float)) else params[i]

class CancelFitException(Exception): pass
class CancelBatchException(Exception): pass
class CancelEpochException(Exception): pass

'''
net: Network model.
n_epoch: Numer of epochs.
opt: Optimizer.
f_loss: Loss.
f_metric: Metric.
dl_train: DataLoader for training.
dl_test: DataLoarder for testing.
tfm_batch: Transformation function for a batch. It should be: (x,y)=tfm_batch(batch).
  where x: input variables given to net, y: target variable.
callbacks: Dictionary of callback functions (function or TFuncList).
lr: Learning rate or list of lr.
device: cpu or cuda.
'''
def Fit(net, n_epoch, opt=None, f_loss=None, f_metric=None,
        dl_train=None, dl_test=None, tfm_batch=None,
        callbacks=None,
        lr=None,
        device=torch.device('cuda')):
  #We use a container to store the  variables to be shared with the callbacks.
  l= TContainer()
  for k,v in locals().items(): l[k]= v
  
  #Default arguments.
  assert(l.opt is not None)
  assert(l.f_loss is not None)

  l.device= torch.device(l.device)  #For type(l.device)==str
  if (l.device=='cuda' or l.device.type=='cuda') and not torch.cuda.is_available():
    l.device= torch.device('cpu')
    print('Fit:WARNING: Device is switched to cpu as cuda is not available.')

  default_callbacks= {e:TFuncList() for e in 
                      ('fit_begin', 'fit_end', 
                       'epoch_train_begin', 'epoch_train_end', 'epoch_test_begin', 'epoch_test_end',
                       'batch_train_begin', 'batch_train_end', 'batch_test_begin', 'batch_test_end',
                       'train_after_prediction', 'test_after_prediction', 'train_after_backward')}
  l.callbacks= MergeDictSum(default_callbacks, l.callbacks, allow_new_key=False) if l.callbacks is not None else default_callbacks

  if l.lr is not None:  AssignParamGroups(l.opt, 'lr', l.lr)

  try:
    l.t_start= time.time()
    l.callbacks['fit_begin'](l)
    for l.i_epoch in range(l.n_epoch):###############
      try:
        if l.dl_train:
          l.callbacks['epoch_train_begin'](l)
          l.sum_loss= 0.0
          l.sum_metric= None
          l.net.train()
          for l.i_batch, l.batch in enumerate(l.dl_train):
            try:
              l.callbacks['batch_train_begin'](l)
              l.opt.zero_grad()
              l.x,l.y_trg,l.pred= PredBatch(l.net, l.batch, tfm_batch=l.tfm_batch, device=l.device)
              l.do_bkw= True
              l.callbacks['train_after_prediction'](l)
              l.loss= l.f_loss(l.pred, l.y_trg)
              if l.do_bkw: l.loss.backward()
              l.do_opt= True
              l.callbacks['train_after_backward'](l)
              if l.do_opt: l.opt.step()
              l.sum_loss+= float(l.loss)
            except CancelBatchException:
              pass
            l.callbacks['batch_train_end'](l)
          l.loss= l.sum_loss/len(l.dl_train)
          l.metric= None
          l.callbacks['epoch_train_end'](l)

        if l.dl_test:
          l.callbacks['epoch_test_begin'](l)
          l.sum_loss= 0.0
          l.sum_metric= 0.0
          l.net.eval()
          with torch.no_grad():
            for l.i_batch, l.batch in enumerate(l.dl_test):
              try:
                l.callbacks['batch_test_begin'](l)
                l.x,l.y_trg,l.pred= PredBatch(l.net, l.batch, tfm_batch=l.tfm_batch, device=l.device)
                l.callbacks['test_after_prediction'](l)
                l.sum_loss+= float(l.f_loss(l.pred, l.y_trg))
                if l.f_metric:  l.sum_metric+= float(l.f_metric(l.pred, l.y_trg))
              except CancelBatchException:
                pass
              l.callbacks['batch_test_end'](l)
          l.loss= l.sum_loss/len(l.dl_test)
          l.metric= None if l.f_metric is None else l.sum_metric/len(l.dl_test)
          l.callbacks['epoch_test_end'](l)
      except CancelEpochException:
        pass
  except CancelFitException:
    pass
  l.callbacks['fit_end'](l)

class TScheduler(object):
  def __init__(self, kind, start, end):
    self.start= start
    self.end= end
    self.f= getattr(self,'f_'+kind)
  def f_lin(self, pos):  return self.start+pos*(self.end-self.start)
  def f_cos(self, pos):  return self.start+(1. + np.cos(np.pi*(1-pos)))*(self.end-self.start)/2.
  def f_exp(self, pos):  return self.start*np.power(self.end/self.start,pos)
  def __call__(self, pos):  return self.f(pos)

'''
Combination of multiple schedulers.
  via_points: List of (pos, value) pairs.
    via_points[0][0] must be 0 and via_points[-1][0] must be 1.
  kinds: Kinds of each section.
'''
class TCmbScheduler(object):
  def __init__(self, via_points, kinds):
    assert(via_points[0][0]==0. and via_points[-1][0]==1.)
    assert(len(via_points)==len(kinds)+1)
    self.via_points= via_points
    self.via_pos= np.array([pos for pos,val in via_points])
    self.schedulers= [TScheduler(k,via_points[i][1],via_points[i+1][1]) for i,k in enumerate(kinds)]
  def __call__(self, pos):
    idx= max(0,min(len(self.via_pos)-2, sum(pos>=self.via_pos)-1))
    pos_sec= (pos-self.via_points[idx][0]) / (self.via_points[idx+1][0]-self.via_points[idx][0])
    return self.schedulers[idx](pos_sec)

class TLRFinder(TCallbacks):
  def __init__(self, start_lr, end_lr, num_iter, r_div):
    self.sch= TFuncList([TScheduler('exp', s, e) for s,e in zip(start_lr,end_lr)]) \
              if isinstance(start_lr,list) else TScheduler('exp', start_lr, end_lr)
    self.i_iter= 0
    self.num_iter= num_iter
    self.r_div= r_div
    self.best_loss= float('inf')
    self.log_loss= []
    self.log_lr= []
  def cb_batch_train_begin(self, l):
    pos= self.i_iter/self.num_iter
    self.log_lr.append(self.sch(pos))
    AssignParamGroups(l.opt, 'lr', self.log_lr[-1])
    if round(pos*100)%20==0:  print(f'FindLR progress: {pos*100}%')
    self.i_iter+= 1
  def cb_batch_train_end(self, l):
    self.log_loss.append(float(l.loss))
    if self.log_loss[-1]<self.best_loss:  self.best_loss= self.log_loss[-1]
    if self.i_iter>self.num_iter:  raise CancelFitException()
    if self.r_div is not None and self.log_loss[-1]>self.r_div*self.best_loss:  raise CancelFitException()
  def cb_fit_begin(self, l):
    self.states= {obj:copy.deepcopy(l[obj].state_dict()) for obj in ('net','opt','f_loss')}
  def cb_fit_end(self, l):
    for obj,st in self.states.items():  l[obj].load_state_dict(st)

def FindLongestDownhill(log_loss):
  l_dwnhill= [1]*len(log_loss)
  for i in range(1,len(log_loss)):
    l_dwnhill[i]= max([l_dwnhill[i]]+[ld+1 for ld,loss in zip(l_dwnhill[:i],log_loss[:i]) if log_loss[i]<loss])
  i_end= l_dwnhill.index(max(l_dwnhill))
  i_start= i_end-l_dwnhill[i_end]
  i_middle= (i_start+i_end)//2
  return i_middle, (i_start, i_end)

def FindLR(net, opt=None, f_loss=None, dl_train=None, tfm_batch=None, device=torch.device('cuda'),
           start_lr=1e-7, end_lr=1, num_iter=100, r_div=None, with_suggest=True, n_filter=10, show_plot=True):
  n_epoch= num_iter//len(dl_train)+1
  lrf= TLRFinder(start_lr, end_lr, num_iter, r_div)
  Fit(net, n_epoch, opt=opt, f_loss=f_loss, f_metric=None,
      dl_train=dl_train, tfm_batch=tfm_batch,
      callbacks=lrf.Callbacks(),
      device=device)
  if with_suggest or show_plot:
    log_loss_filtered= [np.mean(lrf.log_loss[max(0,i+1-n_filter//2):i+1+n_filter//2]) for i in range(len(lrf.log_loss))]
  if with_suggest:
    lr_idx,(lr_idx_min,lr_idx_max)= FindLongestDownhill(log_loss_filtered)
  if show_plot:
    fig= plt.figure()
    ax_loss= fig.add_subplot(1,1,1,title='FindLR',xlabel='lr',ylabel='loss')
    ax_loss.plot(lrf.log_lr, lrf.log_loss, color='blue', linewidth=0.5, label='LR')
    ax_loss.plot(lrf.log_lr, log_loss_filtered, color='green', linewidth=2, label='LR(filtered)')
    if with_suggest:  
      ax_loss.scatter([lrf.log_lr[lr_idx]], [log_loss_filtered[lr_idx]], color='red', label='LR(suggested)')
      ax_loss.scatter([lrf.log_lr[lr_idx_min]], [log_loss_filtered[lr_idx_min]], label='LR(suggested min)')
      ax_loss.scatter([lrf.log_lr[lr_idx_max]], [log_loss_filtered[lr_idx_max]], label='LR(suggested max)')
    ax_loss.set_ylim(bottom=min(lrf.log_loss),top=max(lrf.log_loss[:len(lrf.log_loss)//2]))
    # ax_loss.set_yscale('log')
    ax_loss.set_xscale('log')
    ax_loss.legend()
    plt.show()
  if with_suggest:  return lrf.log_lr[lr_idx], (lrf.log_lr[lr_idx_min], lrf.log_lr[lr_idx_max], (lrf.log_loss, lrf.log_lr))
  return lrf.log_loss, lrf.log_lr

class TOneCycleScheduler(TCallbacks):
  def __init__(self, lr_max, momentums, div_init, div_final, pos_peak, num_iter):
    lr_max= lr_max if isinstance(lr_max,float) else np.array(lr_max)
    self.num_iter= num_iter
    self.sch= {'lr': TCmbScheduler(((0.,lr_max/div_init), (pos_peak,lr_max), (1.,lr_max/div_final)), ('cos','cos'))}
    if momentums is not None:
      self.sch['momentum']= TCmbScheduler(((0.,momentums[0]), (pos_peak,momentums[1]), (1.,momentums[2])), ('cos','cos'))
    self.i_iter= 0
  def cb_batch_train_begin(self, l):
    pos= self.i_iter/self.num_iter
    for key,sch in self.sch.items():
      AssignParamGroups(l.opt, key, sch(pos))
    self.i_iter+= 1

'''
momentums: Tuple of three momentums (init, peak, final); e.g. (0.95,0.85,0.95).
'''
def FitOneCycle(net, n_epoch, opt=None, f_loss=None, f_metric=None,
                dl_train=None, dl_test=None, tfm_batch=None,
                callbacks=None,
                lr_max=None, momentums=None, div_init=25., div_final=1e5, pos_peak=0.25,
                device=torch.device('cuda')):
  assert(opt is not None)
  if lr_max is None:  lr_max= [param_group['lr'] for param_group in opt.param_groups]
  num_iter= n_epoch*len(dl_train)
  ocsch= TOneCycleScheduler(lr_max, momentums, div_init, div_final, pos_peak, num_iter)
  cbs= ocsch.Callbacks()
  callbacks= [callbacks,cbs] if isinstance(callbacks,dict) else (list(callbacks)+[cbs] if callbacks is not None else cbs)
  Fit(net, n_epoch, opt=opt, f_loss=f_loss, f_metric=f_metric,
        dl_train=dl_train, dl_test=dl_test, tfm_batch=tfm_batch,
        callbacks=callbacks, device=device)


# Visualization tools.

def PlotImgGrid(imgs, labels, rows=3, cols=5, labelsize=10, figsize=None, perm_img=True):
  assert(len(imgs)==len(labels))
  rows= min(rows, np.ceil(len(imgs)/cols))
  if figsize is None:  figsize= (12,12/cols*rows)
  fig= plt.figure(figsize=figsize)
  for i,(img,label) in enumerate(zip(imgs,labels)):
    ax= fig.add_subplot(rows, cols, i+1)
    ax.set_title(label, fontsize=labelsize)
    ax.imshow(img.permute(1,2,0) if perm_img else img)
  fig.tight_layout()
  plt.show()


if __name__=='__main__':
  pass
