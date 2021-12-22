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


# General utilities.

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

  def Show(self, mode='all', with_show=True):
    if mode in ('all','summary'):
      print(f'total epochs: {len(self.time_train)}')
      print(f'total time: {sum(self.time_train)/60.:.2f}min')
      if len(self.loss_train)>0:  print(f'best loss(train): {min(self.loss_train)}@{self.loss_train.index(min(self.loss_train))}')
      if len(self.loss_test)>0:  print(f'best loss(test): {min(self.loss_test)}@{self.loss_test.index(min(self.loss_test))}')
      if len(self.metric_train)>0:  print(f'best metric(train): {min(self.metric_train)}@{self.metric_train.index(min(self.metric_train))}')
      if len(self.metric_test)>0:  print(f'best metric(test): {min(self.metric_test)}@{self.metric_test.index(min(self.metric_test))}')
    if mode in ('all','plot'):
      self.Plot(with_show=False)
      self.PlotLR(with_show=False)
      if with_show:  plt.show()

  def Plot(self, with_show=True):
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
    if with_show:  plt.show()

  def PlotLR(self, with_show=True):
    fig= plt.figure()
    ax_lr= fig.add_subplot(1,1,1,title='Learning rate',xlabel='iteration',ylabel='lr')
    ax_lr.plot(range(len(self.lr)), self.lr, color='blue', label='lr')
    ax_lr.set_yscale('log')
    ax_lr.legend()
    if with_show:  plt.show()

class TDisp(TCallbacks):
  def __init__(self):
    self.i_epoch= 0
  def cb_fit_begin(self, l):
    print('i_epoch\tloss(train)\tloss(test)\tmetric(train)\tmetric(test)\t time')
  def cb_epoch_train_begin(self, l):
    self.t0= time.time()
  def cb_epoch_train_end(self, l):
    self.time_train= time.time()-self.t0
    self.loss_train= l.loss if l.loss is not None else np.nan
    self.metric_train= l.metric if l.metric is not None else np.nan
  def cb_epoch_test_begin(self, l):
    self.t0= time.time()
  def cb_epoch_test_end(self, l):
    self.time_test= time.time()-self.t0
    self.loss_test= l.loss if l.loss is not None else np.nan
    self.metric_test= l.metric if l.metric is not None else np.nan
    print(f'{self.i_epoch}\t{self.loss_train:.8f}\t{self.loss_test:.8f}\t{self.metric_train:.8f}\t{self.metric_test:.8f}\t {self.time_train+self.time_test:.6f}')
    self.i_epoch+= 1


# Learning utilities.

def FindDevice(device=torch.device('cuda')):
  device= torch.device(device)  #For type(l.device)==str
  if (device=='cuda' or device.type=='cuda') and not torch.cuda.is_available():
    device= torch.device('cpu')
    print('WARNING: Device is switched to cpu as cuda is not available.')
  return device

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
Calculate average loss f_loss (or metric) for a dataset (dset) or a data-loader (dl).
'''
def EvalLoss(net, f_loss=None, dl=None, dset=None, tfm_batch=None, device=torch.device('cuda'), dl_args=None):
  assert((dl is None)!=(dset is None))
  device= FindDevice(device)
  if dset is not None:
    default_dl_args= dict(batch_size=64, shuffle=False, num_workers=2)
    dl_args= MergeDict(default_dl_args,dl_args) if dl_args else default_dl_args
    dl= torch.utils.data.DataLoader(dataset=dset, **dl_args)
  net.eval()
  with torch.no_grad():
    loss= sum((float(f_loss(*reversed(PredBatch(net, batch, tfm_batch=tfm_batch, device=device, with_x=False))))
                for batch in dl)) / len(dl)
  return loss

'''
Helper to assign parameters.
'''
def AssignParamGroups(obj, key, params):
  for i, param_group in enumerate(obj.param_groups):
    param_group[key]= params if isinstance(params,(int,float)) else params[i]

'''
Freeze parameters except for sub-network in unfrozen.
'''
def FreezeParametersExceptFor(net, unfrozen):
  ids_unfrozen= [id(p) for pg in unfrozen for p in pg.parameters()] if isinstance(unfrozen,(tuple,list)) else [id(p) for p in unfrozen.parameters()]
  for p in net.parameters():
    p.requires_grad= id(p) in ids_unfrozen
'''
Unfreeze all parameters.
'''
def UnfreezeAllParameters(net):
  for p in net.parameters():
    p.requires_grad= True

class CancelFitException(Exception): pass
class CancelBatchException(Exception): pass
class CancelEpochException(Exception): pass

'''
Save state_dict of net, opt, f_loss into a destination dst.
dst can be a dict (deep-copied) or a file (saved).
'''
def SaveStateDict(dst, net=None, opt=None, f_loss=None):
  states= {
    'net': net.state_dict() if hasattr(net,'state_dict') else None,
    'opt': opt.state_dict() if hasattr(opt,'state_dict') else None,
    'f_loss': f_loss.state_dict() if hasattr(f_loss,'state_dict') else None,
    }
  if isinstance(dst,dict):
    for obj,st in states.items():
      dst[obj]= copy.deepcopy(st)
  elif isinstance(dst,str):
    torch.save(states, dst)
  else:
    raise Exception(f'SaveStateDict: unrecognized destination type: {type(dst)}')

'''
Load state_dict of net, opt, f_loss from a source src.
src can be a dict or a file.
'''
def LoadStateDict(src, net=None, opt=None, f_loss=None, device=None, strict=True, with_exception=False):
  if device is None:  device= 'cpu'
  if isinstance(src,dict):
    states= src
  elif isinstance(src,str):
    states= torch.load(src, map_location=device)
  else:
    raise Exception(f'LoadStateDict: unrecognized source type: {type(src)}')
  if net is not None:  net.to(device)
  dst= dict(net=(net,dict(strict=strict)), opt=(opt,dict()), f_loss=(f_loss,dict()))
  for obj in dst.keys():
    if dst[obj][0] is not None:  
      if obj in states and hasattr(dst[obj][0],'load_state_dict'):  
        dst[obj][0].load_state_dict(states[obj], **dst[obj][1])
      else:  
        print(f'LoadStateDict:WARNING: failed to load "{obj}".')
        print(f'  {obj} in states: {obj in states}')
        print(f'  hasattr({obj},load_state_dict): {hasattr(dst[obj][0],"load_state_dict")}')
        if with_exception:
          raise Exception(f'LoadStateDict: failed to load "{obj}"')
  

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

  l.device= FindDevice(l.device)

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
          l.sum_metric= 0.0
          l.net.train()
          for l.i_batch, l.batch in enumerate(l.dl_train):
            l.forward_value_error= True
            l.value_error= None
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
              if l.f_metric:
                with torch.no_grad():  l.sum_metric+= float(l.f_metric(l.pred, l.y_trg))
            except CancelBatchException:
              pass
            except ValueError as e:
              l.value_error= e
              if l.forward_value_error:  raise e
            l.callbacks['batch_train_end'](l)
          l.loss= l.sum_loss/len(l.dl_train)
          l.metric= None if l.f_metric is None else l.sum_metric/len(l.dl_train)
          l.callbacks['epoch_train_end'](l)

        if l.dl_test:
          l.callbacks['epoch_test_begin'](l)
          l.sum_loss= 0.0
          l.sum_metric= 0.0
          l.net.eval()
          with torch.no_grad():
            for l.i_batch, l.batch in enumerate(l.dl_test):
              l.forward_value_error= True
              l.value_error= None
              try:
                l.callbacks['batch_test_begin'](l)
                l.x,l.y_trg,l.pred= PredBatch(l.net, l.batch, tfm_batch=l.tfm_batch, device=l.device)
                l.callbacks['test_after_prediction'](l)
                l.sum_loss+= float(l.f_loss(l.pred, l.y_trg))
                if l.f_metric:  l.sum_metric+= float(l.f_metric(l.pred, l.y_trg))
              except CancelBatchException:
                pass
              except ValueError as e:
                l.value_error= e
                if l.forward_value_error:  raise e
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
    l.forward_value_error= False
    pos= self.i_iter/self.num_iter
    self.log_lr.append(self.sch(pos))
    AssignParamGroups(l.opt, 'lr', self.log_lr[-1])
    if round(pos*100)%20==0:  print(f'FindLR progress: {pos*100}%')
    self.i_iter+= 1
  def cb_batch_train_end(self, l):
    if l.value_error is not None:
      self.log_lr.pop(-1)
      print('FindLR is terminated due to a ValueError')
      raise CancelFitException()
    self.log_loss.append(float(l.loss))
    if self.log_loss[-1]<self.best_loss:  self.best_loss= self.log_loss[-1]
    if self.i_iter>self.num_iter:  raise CancelFitException()
    if self.r_div is not None and self.log_loss[-1]>self.r_div*self.best_loss:  raise CancelFitException()
  def cb_fit_begin(self, l):
    self.states= {}
    SaveStateDict(self.states, net=l.net, opt=l.opt, f_loss=l.f_loss)
  def cb_fit_end(self, l):
    LoadStateDict(self.states, net=l.net, opt=l.opt, f_loss=l.f_loss, device=l.device, with_exception=True)

def FindLongestDownhill(log_loss):
  l_dwnhill= [1]*len(log_loss)
  for i in range(1,len(log_loss)):
    l_dwnhill[i]= max([l_dwnhill[i]]+[ld+1 for ld,loss in zip(l_dwnhill[:i],log_loss[:i]) if log_loss[i]<loss])
  i_end= l_dwnhill.index(max(l_dwnhill))
  i_start= i_end-l_dwnhill[i_end]
  i_middle= (i_start+i_end)//2
  return i_middle, (i_start, i_end)

def FindLR(net, opt=None, f_loss=None, dl_train=None, tfm_batch=None, device=torch.device('cuda'),
           start_lr=1e-7, end_lr=1, num_iter=100, r_div=None, with_suggest=True, n_filter=20, show_plot=True):
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


# Network modules.

'''
Do nothing function.
'''
def Noop(x, *args, **kwargs):
  return x

'''
Do nothing module.
'''
class TNoop(torch.nn.Module):
  def __init__(self, *args, **kwargs):
    super(TNoop,self).__init__()
  def forward(self, x):
    return x

'''
Create a convolutional layer optionally with ReLu and normalization layers.
Ref. https://github.com/fastai/fastai/tree/master/fastai/layers.py
In transpose case, padding and output_padding are automatically computed if they are None 
so that the transposed conv is the exact reverse in terms of the shape as conv 
with the same kernel_size and stride.
'''
def ConvLayer(in_channels, out_channels, kernel_size, stride=1, padding=None,
              bias=None, ndim=2, norm_type='batch', batchnorm_first=True,
              activation=torch.nn.ReLU, transpose=False, init='auto', bias_std=0.01, **kwargs):
  if not transpose:
    if padding is None: padding= (kernel_size-1)//2
  else:
    if padding is None:
      if kwargs.get('output_padding') is not None:
        output_padding= kwargs['output_padding']
        padding= (kernel_size-stride+output_padding)//2
      else:
        if stride>kernel_size:
          output_padding= stride-kernel_size
          padding= 0
        else:
          output_padding= (kernel_size-stride)%2
          padding= (kernel_size-stride+output_padding)//2
    else:
      output_padding= kwargs.get('output_padding', stride-kernel_size+2*padding)
    kwargs['output_padding']= output_padding
  bn= norm_type in ('batch', 'batch_zero')
  inn= norm_type in ('instance', 'instance_zero')
  if bias is None: bias= not (bn or inn)
  conv_func= getattr(torch.nn, f'Conv{"Transpose" if transpose else ""}{ndim}d')
  conv= conv_func(in_channels, out_channels, kernel_size=kernel_size, 
                  stride=stride, padding=padding, bias=bias, **kwargs)
  act= (None if activation is None else 
        activation(inplace=True) if activation in (torch.nn.ReLU,torch.nn.ReLU6,torch.nn.LeakyReLU) else
        activation())
  if getattr(conv,'bias',None) is not None and bias_std is not None:
    if bias_std!=0: torch.nn.init.normal_(conv.bias, 0.0, bias_std)
    else: conv.bias.data.zero_()
  f_init= None
  if act is not None and init=='auto':
    if hasattr(act.__class__, '__default_init__'):
      f_init= act.__class__.__default_init__
    else:  f_init= getattr(act, '__default_init__', None)
    if f_init is None and act in (torch.nn.ReLU,torch.nn.ReLU6,torch.nn.LeakyReLU):
      f_init= torch.nn.init.kaiming_uniform_
  if f_init is not None: f_init(conv.weight)
  if   norm_type=='weight':   conv= torch.nn.utils.weight_norm(conv)
  elif norm_type=='spectral': conv= torch.nn.utils.spectral_norm(conv)
  layers= [conv]
  act_bn= []
  if act is not None: act_bn.append(act)
  if bn: 
    bnl= getattr(torch.nn, f'BatchNorm{ndim}d')(out_channels)
    if bnl.affine:
      bnl.bias.data.fill_(1e-3)
      bnl.weight.data.fill_(0. if norm_type=='batch_zero' else 1.)
    act_bn.append(bnl)
  if inn: 
    innl= getattr(torch.nn, f'InstanceNorm{ndim}d')(out_channels, affine=True)
    if innl.affine:
      innl.bias.data.fill_(1e-3)
      innl.weight.data.fill_(0. if norm_type=='instance_zero' else 1.)
    act_bn.append(innl)
  if batchnorm_first: act_bn.reverse()
  layers+= act_bn
  return torch.nn.Sequential(*layers)

'''
Simple self attention.
Ref. https://github.com/fastai/fastai/blob/master/fastai/layers.py
'''
class TSimpleSelfAttention(torch.nn.Module):
  def __init__(self, in_channels, kernel_size=1, symmetric=False):
    super(TSimpleSelfAttention,self).__init__()
    self.symmetric,self.in_channels= symmetric,in_channels
    conv= torch.nn.Conv1d(in_channels, in_channels, kernel_size, stride=1, padding=kernel_size//2, bias=False)
    torch.nn.init.kaiming_normal_(conv.weight)
    self.conv= torch.nn.utils.spectral_norm(conv)
    self.gamma= torch.nn.Parameter(torch.tensor([0.]))

  def forward(self, x):
    if self.symmetric:
      c= self.conv.weight.view(self.in_channels,self.in_channels)
      c= (c + c.t())/2.0
      self.conv.weight= c.view(self.in_channels,self.in_channels,1)
    size= x.size()
    x= x.view(*size[:2],-1)
    convx= self.conv(x)
    xxT= torch.bmm(x,x.permute(0,2,1).contiguous())
    o= self.gamma * torch.bmm(xxT, convx) + x
    return o.view(*size).contiguous()

class TResBlockReduction(torch.nn.Module):
  def __init__(self, in_channels, reduction, activation, ndim):
    super(TResBlockReduction,self).__init__()
    nf= np.ceil(in_channels//reduction/8)*8
    self.red= torch.nn.Sequential(
              getattr(torch.nn, f'AdaptiveAvgPool{ndim}d')(output_size=1),
              ConvLayer(in_channels, nf, kernel_size=1, norm_type=None, activation=activation),
              ConvLayer(nf, in_channels, kernel_size=1, norm_type=None, activation=torch.nn.Sigmoid))
  def forward(self, x):
    return x * self.red(x)

'''
ResNet block.
Ref. https://github.com/fastai/fastai/blob/master/fastai/layers.py
In transpose case, upsample is added to the idpath.
'''
class TResBlock(torch.nn.Module):
  def __init__(self, expansion, in_channels, out_channels, stride=1, 
               groups=1, reduction=None, hidden1_channels=None, hidden2_channels=None, 
               depthwise_conv=False, groups2=1, self_attention=False, sa_symmetric=False, 
               norm_type='batch', activation=torch.nn.ReLU, ndim=2, kernel_size=3,
               pool=None, pool_first=True, upsample=torch.nn.Upsample, upsample_first=False, **kwargs):
    super(TResBlock,self).__init__()
    norm2= ('batch_zero' if norm_type=='batch' else
            'instance_zero' if norm_type=='instance' else norm_type)
    if hidden2_channels is None: hidden2_channels= out_channels
    if hidden1_channels is None: hidden1_channels= hidden2_channels
    out_channels,in_channels= out_channels*expansion,in_channels*expansion
    kwargs1= dict(norm_type=norm_type, activation=activation, ndim=ndim, **kwargs)
    kwargs2= dict(norm_type=norm2, activation=None, ndim=ndim, **kwargs)
    if expansion==1:
      convpath= [ConvLayer(in_channels, hidden2_channels, kernel_size, stride=stride, groups=in_channels if depthwise_conv else groups, **kwargs1),
                 ConvLayer(hidden2_channels, out_channels, kernel_size, groups=groups2, **kwargs2)]
    else: 
      convpath= [ConvLayer(in_channels, hidden1_channels, 1, **kwargs1),
                 ConvLayer(hidden1_channels, hidden2_channels, kernel_size, stride=stride, groups=hidden1_channels if depthwise_conv else groups, **kwargs1),
                 ConvLayer(hidden2_channels, out_channels, 1, groups=groups2, **kwargs2)]
    if reduction:  convpath.append(TResBlockReduction(out_channels, reduction=reduction, activation=activation, ndim=ndim))
    if self_attention:  convpath.append(TSimpleSelfAttention(out_channels, kernel_size=1, symmetric=sa_symmetric))
    self.convpath= torch.nn.Sequential(*convpath)
    idpath= []
    if in_channels!=out_channels: idpath.append(ConvLayer(in_channels, out_channels, kernel_size=1, activation=None, ndim=ndim, **kwargs))
    if stride!=1: 
      if not kwargs.get('transpose',False):
        if pool is None:  pool= getattr(torch.nn, f'AvgPool{ndim}d')
        idpath.insert((1,0)[pool_first], pool(kernel_size=stride, stride=None, padding=0, ceil_mode=True))
      else:
        idpath.insert((1,0)[upsample_first], upsample(scale_factor=stride, mode='nearest'))
    self.idpath= torch.nn.Sequential(*idpath)
    self.act= activation(inplace=True) if activation in (torch.nn.ReLU,torch.nn.ReLU6,torch.nn.LeakyReLU) else activation()

  def forward(self, x): 
    return self.act(self.convpath(x) + self.idpath(x))

def InitCNN(m):
  if getattr(m, 'bias', None) is not None:  torch.nn.init.constant_(m.bias, 0)
  if isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear)):  torch.nn.init.kaiming_normal_(m.weight)
  for l in m.children(): InitCNN(l)

class TResNet(torch.nn.Sequential):
  def __init__(self, block, expansion, layers, p_dropout=0.0, in_channels=3, out_channels=1000, stem_sizes=(32,32,64),
               widen=1.0, with_fc=True, self_attention=False, activation=torch.nn.ReLU, ndim=2, kernel_size=3, 
               stride=2, stem_stride=None, rnpool=None, rnpool_stride=None, **kwargs):
    self.block       = block      
    self.expansion   = expansion  
    self.activation  = activation 
    self.ndim        = ndim       
    self.kernel_size = kernel_size
    if kernel_size%2==0:  raise Exception('kernel size has to be odd!')
    if stem_stride is None:  stem_stride= stride
    if rnpool_stride is None:  rnpool_stride= stride

    stem= self.make_stem(in_channels, stem_sizes, stem_stride)
    block_sizes= [int(o*widen) for o in [64,128,256,512] +[256]*(len(layers)-4)]
    block_sizes= [64//expansion] + block_sizes
    blocks= self.make_blocks(layers, block_sizes, self_attention, stride, **kwargs)
    if rnpool is None:  rnpool= getattr(torch.nn, f"MaxPool{ndim}d")
    pool= rnpool(kernel_size=kernel_size, stride=rnpool_stride, padding=kernel_size//2)

    if with_fc:
      super(TResNet,self).__init__(
            *stem, pool, *blocks,
            getattr(torch.nn, f'AdaptiveAvgPool{ndim}d')(output_size=1), 
            torch.nn.Flatten(), 
            torch.nn.Dropout(p_dropout),
            torch.nn.Linear(block_sizes[-1]*expansion, out_channels),
            )
      self.out_channels= out_channels
    else:
      super(TResNet,self).__init__(*stem, pool, *blocks)
      self.out_channels= block_sizes[len(layers)]
    InitCNN(self)

  def make_stem(self, in_channels, stem_sizes, stride):
    stem_sizes= [in_channels, *stem_sizes]
    return [ConvLayer(stem_sizes[i], stem_sizes[i+1], kernel_size=self.kernel_size, 
                       stride=(1 if i==0 else stride) if isinstance(stride,int) else stride[i],
                       activation=self.activation, ndim=self.ndim)
             for i in range(len(stem_sizes)-1)]

  def make_blocks(self, layers, block_sizes, self_attention, stride, **kwargs):
    return [self.make_layer(ni=block_sizes[i], nf=block_sizes[i+1], n_blocks=l,
                            stride=(1 if i==0 else stride) if isinstance(stride,int) else stride[i], 
                            self_attention=self_attention and i==len(layers)-4, **kwargs)
            for i,l in enumerate(layers)]

  def make_layer(self, ni, nf, n_blocks, stride, self_attention, **kwargs):
    return torch.nn.Sequential(
          *[self.block(self.expansion, ni if i==0 else nf, nf, stride=(1 if i==0 else stride) if isinstance(stride,int) else stride[i],
                    self_attention=self_attention and i==(n_blocks-1), activation=self.activation, ndim=self.ndim, kernel_size=self.kernel_size, **kwargs)
            for i in range(n_blocks)])

def TResNet18 (**kwargs): return TResNet(TResBlock, 1, [2, 2,  2, 2], **kwargs)
def TResNet34 (**kwargs): return TResNet(TResBlock, 1, [3, 4,  6, 3], **kwargs)
def TResNet50 (**kwargs): return TResNet(TResBlock, 4, [3, 4,  6, 3], **kwargs)
def TResNet101(**kwargs): return TResNet(TResBlock, 4, [3, 4, 23, 3], **kwargs)
def TResNet152(**kwargs): return TResNet(TResBlock, 4, [3, 8, 36, 3], **kwargs)
def TResNet18_deep  (**kwargs): return TResNet(TResBlock, 1, [2,2,2,2,1,1], **kwargs)
def TResNet34_deep  (**kwargs): return TResNet(TResBlock, 1, [3,4,6,3,1,1], **kwargs)
def TResNet50_deep  (**kwargs): return TResNet(TResBlock, 4, [3,4,6,3,1,1], **kwargs)
def TResNet18_deeper(**kwargs): return TResNet(TResBlock, 1, [2,2,1,1,1,1,1,1], **kwargs)
def TResNet34_deeper(**kwargs): return TResNet(TResBlock, 1, [3,4,6,3,1,1,1,1], **kwargs)
def TResNet50_deeper(**kwargs): return TResNet(TResBlock, 4, [3,4,6,3,1,1,1,1], **kwargs)


'''
Make a dense (fully-connected linear) layer optionally with a normalization and an activation layers.
'''
def DenseLayer(in_channels, out_channels,
              bias=True, norm_type='batch', batchnorm_first=True,
              activation=torch.nn.LeakyReLU, init='auto', bias_std=0.5):
  bn= norm_type in ('batch', 'batch_zero')
  dense= torch.nn.Linear(in_channels, out_channels, bias=bias)
  act= (None if activation is None else 
        activation(inplace=True) if activation in (torch.nn.ReLU,torch.nn.ReLU6,torch.nn.LeakyReLU) else
        activation())
  if getattr(dense,'bias',None) is not None and bias_std is not None:
    if bias_std!=0: torch.nn.init.normal_(dense.bias, 0.0, bias_std)
    else: dense.bias.data.zero_()
  f_init= None
  if act is not None and init=='auto':
    if hasattr(act.__class__, '__default_init__'):
      f_init= act.__class__.__default_init__
    else:  f_init= getattr(act, '__default_init__', None)
    if f_init is None and act in (torch.nn.ReLU,torch.nn.ReLU6,torch.nn.LeakyReLU):
      f_init= torch.nn.init.xavier_uniform_
  if f_init is not None: f_init(dense.weight,gain=torch.nn.init.calculate_gain('leaky_relu'))
  if   norm_type=='weight':   dense= torch.nn.utils.weight_norm(dense)
  elif norm_type=='spectral': dense= torch.nn.utils.spectral_norm(dense)
  layers= [dense]
  act_bn= []
  if act is not None: act_bn.append(act)
  if bn:
    bnl= torch.nn.BatchNorm1d(out_channels)
    if bnl.affine:
      bnl.bias.data.fill_(1e-3)
      bnl.weight.data.fill_(0. if norm_type=='batch_zero' else 1.)
    act_bn.append(bnl)
  if batchnorm_first: act_bn.reverse()
  layers+= act_bn
  return torch.nn.Sequential(*layers)

'''
ResNet block of dense network.
ref. https://www.mdpi.com/1099-4300/22/2/193/pdf
'''
class TResDenseBlock(torch.nn.Module):
  def __init__(self, in_channels, out_channels, hidden_channels=None,
               activation=torch.nn.LeakyReLU, **kwargs):
    super(TResDenseBlock,self).__init__()
    if hidden_channels is None: hidden_channels= out_channels
    densepath= [DenseLayer(in_channels, hidden_channels, activation=activation, **kwargs),
                DenseLayer(hidden_channels, out_channels, activation=None, **kwargs)]
    self.densepath= torch.nn.Sequential(*densepath)
    idpath= []
    if in_channels!=out_channels: idpath.append(DenseLayer(in_channels, out_channels, activation=None, **kwargs))
    self.idpath= torch.nn.Sequential(*idpath)
    self.act= activation(inplace=True) if activation in (torch.nn.ReLU,torch.nn.ReLU6,torch.nn.LeakyReLU) else activation()

  def forward(self, x): 
    return self.act(self.densepath(x) + self.idpath(x))


'''
ResNet in inverse order.
'''
class TResNetDecoder(torch.nn.Sequential):
  def __init__(self, block, expansion, layers, p_dropout=0.0, in_channels=100, out_imgshape=(3,32,32),
               first_imgshape=(1,1), stem_sizes=(64,32,32), upsample_method='upsample',
               widen=1.0, self_attention=False, activation=torch.nn.ReLU, ndim=2, kernel_size=3, 
               stride=2, stem_stride=None, **kwargs):
    self.block       = block
    self.expansion   = expansion
    self.upsample_method = upsample_method
    self.activation  = activation
    self.ndim        = ndim
    self.kernel_size = kernel_size
    if kernel_size%2==0:  raise Exception('kernel size has to be odd!')
    if stem_stride is None:  stem_stride= stride

    block_sizes= [int(o*widen) for o in [256]*(len(layers)-4) + [512,256,128,64]]
    block_sizes= block_sizes + [64//expansion]
    blocks= self.make_blocks(layers, block_sizes, self_attention, stride, **kwargs)
    stem= self.make_stem(block_sizes[-1]*expansion, stem_sizes, stem_stride)

    super(TResNetDecoder,self).__init__(
          torch.nn.Linear(in_channels, block_sizes[0]*expansion*first_imgshape[0]*first_imgshape[1]), 
          torch.nn.Unflatten(1,(block_sizes[0]*expansion,first_imgshape[0],first_imgshape[1])),
          *blocks,
          # ConvLayer(block_sizes[-1]*expansion, out_imgshape[0], kernel_size=1, activation=None, ndim=ndim),
          *stem,
          ConvLayer(stem_sizes[-1], out_imgshape[0], kernel_size=1, activation=None, ndim=ndim),
          torch.nn.Upsample(size=out_imgshape[1:]),
          )
    InitCNN(self)

  def make_blocks(self, layers, block_sizes, self_attention, stride, **kwargs):
    return [self.make_layer(ni=block_sizes[i], nf=block_sizes[i+1], n_blocks=l,
                            stride=(1 if i==len(layers)-1 else stride) if isinstance(stride,int) else stride[i], 
                            self_attention=self_attention and i==3, **kwargs)
            for i,l in enumerate(layers)]

  def make_layer(self, ni, nf, n_blocks, stride, self_attention, **kwargs):
    if self.upsample_method=='transpose':
      # Using transpose to upsample
      return torch.nn.Sequential(
            *[self.block(self.expansion, ni, nf if i==(n_blocks-1) else ni, stride=(1 if i==(n_blocks-1) else stride) if isinstance(stride,int) else stride[i],
                      self_attention=self_attention and i==0, activation=self.activation, ndim=self.ndim, kernel_size=self.kernel_size, transpose=True, **kwargs)
              for i in range(n_blocks)])
    elif self.upsample_method=='upsample':
      # Using Upsample to upsample
      return torch.nn.Sequential(
              *sum([[self.block(self.expansion, ni, nf if i==(n_blocks-1) else ni, stride=1,
                                self_attention=self_attention and i==0, activation=self.activation, ndim=self.ndim, kernel_size=self.kernel_size,
                                transpose=False, **kwargs),
                     torch.nn.Upsample(scale_factor=(1 if i==(n_blocks-1) else stride) if isinstance(stride,int) else stride[i])
                     ] for i in range(n_blocks)],[]) )

  def make_stem(self, in_channels, stem_sizes, stride):
    stem_sizes= [in_channels, *stem_sizes]
    if self.upsample_method=='transpose':
      # Using transpose to upsample
      stem= [ConvLayer(stem_sizes[i], stem_sizes[i+1], kernel_size=self.kernel_size, 
                       stride=(1 if i==(len(stem_sizes)-1) else stride) if isinstance(stride,int) else stride[i],
                       activation=(None if i==(len(stem_sizes)-1) else self.activation), ndim=self.ndim, transpose=True)
               for i in range(len(stem_sizes)-1)]
    elif self.upsample_method=='upsample':
      # Using Upsample to upsample
      stem= sum([[ConvLayer(stem_sizes[i], stem_sizes[i+1], kernel_size=self.kernel_size, stride=1,
                            activation=(None if i==(len(stem_sizes)-1) else self.activation), ndim=self.ndim,
                            transpose=False),
                  torch.nn.Upsample(scale_factor=(1 if i==(len(stem_sizes)-1) else stride) if isinstance(stride,int) else stride[i])
                  ] for i in range(len(stem_sizes)-1)],[])
    return stem

def TResNet18Decoder(**kwargs): return TResNetDecoder(TResBlock, 1, [2, 2, 2, 2], **kwargs)


# Visualization tools.

def Summary(net, input_size=None, input_data=None, **kwargs):
  default_kwargs= dict(depth=7, row_settings=['var_names'], col_names=['input_size','output_size','num_params'])
  kwargs= MergeDict(default_kwargs, kwargs)
  if input_size is not None:  print(f'input_size={input_size}')
  if input_data is not None:  print(f'input_data.shape={input_data.shape}')
  return torchinfo.summary(net, input_size=input_size, input_data=input_data, **kwargs)

def PlotImgGrid(imgs, labels, rows=3, cols=5, labelsize=10, figsize=None, perm_img=True):
  assert(len(imgs)==len(labels))
  rows= min(rows, np.ceil(len(imgs)/cols))
  if figsize is None:  figsize= (12,12/cols*rows)
  fig= plt.figure(figsize=figsize)
  for i,(img,label) in enumerate(zip(imgs,labels)):
    if i+1>rows*cols:  break
    ax= fig.add_subplot(rows, cols, i+1)
    ax.set_title(label, fontsize=labelsize)
    if img.shape[0]==1:
      ax.imshow(np.repeat(img,3,axis=0).permute(1,2,0) if perm_img else img)
    else:
      ax.imshow(img.permute(1,2,0) if perm_img else img)
  fig.tight_layout()
  plt.show()

def HStackImages(*imgs,margin=1):
  if len(imgs)==0:  return None
  height= max(img.shape[1] for img in imgs)
  width= sum(img.shape[2] for img in imgs)+margin*(len(imgs)-1)
  catimg= torch.zeros((imgs[0].shape[0],height,width),dtype=imgs[0].dtype)
  x= 0
  for img in imgs:
    catimg[:,:img.shape[1],x:x+img.shape[2]]= img
    x+= img.shape[2]+margin
  return catimg

def VStackImages(*imgs,margin=1):
  if len(imgs)==0:  return None
  height= sum(img.shape[1] for img in imgs)+margin*(len(imgs)-1)
  width= max(img.shape[2] for img in imgs)
  catimg= torch.zeros((imgs[0].shape[0],height,width),dtype=imgs[0].dtype)
  y= 0
  for img in imgs:
    catimg[:,y:y+img.shape[1],:img.shape[2]]= img
    y+= img.shape[1]+margin
  return catimg


if __name__=='__main__':
  pass
