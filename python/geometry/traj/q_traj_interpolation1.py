#!/usr/bin/python3
#\file    q_traj_interpolation1.py
#\brief   Joint angle trajectory interpolation.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.19, 2025
import numpy as np
import itertools

def QTrajInterpolation_v0(q_traj, N_int):
  q_traj_int= [q_traj[0]] + sum([(q1+(np.array(q2)-q1)*np.linspace(0,1,N_int+1)[1:].reshape(-1,1)).tolist() for q1,q2 in zip(q_traj[:-1],q_traj[1:])], [])
  return q_traj_int

#Interpolate a joint angle trajectory (list of joint angle vector (list of float))
#  where each interval is divided into fixed number of sub-steps (N) of the same interval.
def QTrajInterpolationByFixedNum(q_traj, N):
  def sub_interpolate(q1, q2, N):
    q1, q2 = np.array(q1), np.array(q2)
    return (q1 + (q2 - q1) * np.linspace(0, 1, N + 1)[1:, None]).tolist()
  q_traj_array= [np.array(q) for q in q_traj]
  interpolated= itertools.chain.from_iterable(
    [sub_interpolate(q1, q2, N) for q1, q2 in zip(q_traj_array[:-1], q_traj_array[1:])])
  return [q_traj[0]] + list(interpolated)

#Interpolate a joint angle trajectory (list of joint angle vector (list of float))
#  where each interval is divided so that a maximum difference of a joint angle is less than dq_max.
def QTrajInterpolationByInterval(q_traj, dq_max):
  def sub_interpolate(q1, q2, N_int):
    q1, q2= np.array(q1), np.array(q2)
    if N_int <= 1:
      return [q2.tolist()]
    return (q1 + (q2 - q1) * np.linspace(0, 1, N_int + 1)[1:, None]).tolist()
  q_traj_array= np.array(q_traj)
  max_dq= np.max(np.abs(np.diff(q_traj_array, axis=0)), axis=1)
  N_int_array= np.ceil(max_dq / dq_max).astype(int)
  #print(f'N_int_array={N_int_array}')
  interpolated= itertools.chain.from_iterable(
    sub_interpolate(q_traj_array[idx], q_traj_array[idx + 1], N_int)
    for idx, N_int in enumerate(N_int_array))
  return [q_traj[0]] + list(interpolated)



import matplotlib.pyplot as plt
import time

# Visualization test function
def plot_q_traj(title, q_traj, interpolated_traj):
  q_traj, interpolated_traj = np.array(q_traj), np.array(interpolated_traj)
  num_joints = q_traj.shape[1]

  interval_lengths = []
  interpolated_idx = 0
  for i in range(len(q_traj) - 1):
    segment_length = 1
    while interpolated_idx + segment_length < len(interpolated_traj) and not np.allclose(interpolated_traj[interpolated_idx + segment_length], q_traj[i + 1]):
      segment_length += 1
    interval_lengths.append(segment_length)
    interpolated_idx += segment_length

  indices_original = [0] + np.cumsum(interval_lengths).tolist()

  fig, axes = plt.subplots(num_joints, 1, figsize=(10, 2*num_joints), sharex=True)
  if num_joints == 1:
    axes = [axes]

  fig.suptitle(title)

  for i, ax in enumerate(axes):
    ax.plot(interpolated_traj[:, i], 'b.-', label='Interpolated')
    ax.plot(indices_original, q_traj[:, i], 'ro', label='Original Points')
    ax.set_ylabel(f'Joint {i+1}')
    ax.grid(True)
    if i == 0:
      ax.legend()
  plt.xlabel('Interpolation Steps')
  plt.tight_layout(rect=[0, 0, 1, 0.96])
  plt.show()

if __name__=='__main__':
  def do_test(q_traj, N_int, dq_max):
    print('-'*15)
    print(f'''Input q_traj={q_traj}
      N_int={N_int}
      dq_max={dq_max}''')

    #start_time= time.time()
    #q_traj_int= QTrajInterpolation_v0(q_traj, N_int)
    #elapsed_time= time.time() - start_time
    #print(f'elapsed_time(v0)={elapsed_time*1000}ms')
    #print(f'len(q_traj_int)(v0)={len(q_traj_int)}')
    ##print(f'q_traj_int(v0)={q_traj_int}')
    #plot_q_traj('v0', q_traj, q_traj_int)

    start_time= time.time()
    q_traj_int= QTrajInterpolationByFixedNum(q_traj, N_int)
    elapsed_time= time.time() - start_time
    print(f'elapsed_time(ByFixedNum)={elapsed_time*1000}ms')
    print(f'len(q_traj_int)(ByFixedNum)={len(q_traj_int)}')
    #print(f'q_traj_int(ByFixedNum)={q_traj_int}')
    plot_q_traj('ByFixedNum', q_traj, q_traj_int)

    start_time= time.time()
    q_traj_int= QTrajInterpolationByInterval(q_traj, dq_max)
    elapsed_time= time.time() - start_time
    print(f'elapsed_time(ByInterval)={elapsed_time*1000}ms')
    print(f'len(q_traj_int)(ByInterval)={len(q_traj_int)}')
    #print(f'q_traj_int(ByInterval)={q_traj_int}')
    plot_q_traj('ByInterval', q_traj, q_traj_int)

  q_traj= [[0, 0], [1, 1], [1.1, 1.1], [3, 3]]
  do_test(q_traj, 5, 0.5)

  q_traj= [[0], [2], [2.1], [5]]
  do_test(q_traj, 5, 0.7)

  q_traj= [[0, 0, 0], [0.2, 0.2, 0.2], [1, 1, 1]]
  do_test(q_traj, 5, 0.25)

  q_traj=[[0.0, -0.12994734942913055, -0.5056460499763489, 0.0, -0.82935631275177, 0.0], [0.04121739564217578, 0.9184917589172862, 1.0526637085895567, 0.0090158708385253, -1.4595433771094193, 0.3111202095289816], [0.04063553856449946, 0.9382126978919846, 0.9672753358914508, 0.009029287318494594, -1.354433110724544, 0.31074720712274884], [0.04048872421933579, 0.9468912054365552, 0.9521502933486955, 0.009042633375871014, -1.3306292060106601, 0.3106770940635928], [0.04034309271255899, 0.9567850318754075, 0.9393422508553467, 0.009058898121564038, -1.307926948867781, 0.3106153110185364], [0.040198629883095284, 0.9678113031026914, 0.9287015902173217, 0.009077603734907473, -1.286259598388362, 0.31056122229799077], [0.04005532173674163, 0.979905447497091, 0.9201157703196435, 0.009098297346674321, -1.2655791873180724, 0.31051439579435935], [0.03991315450844621, 0.9930167610864072, 0.9135003615055642, 0.009120538516093886, -1.2458519914917328, 0.31047455775216337]]
  do_test(q_traj, 5, 0.25)

  q_traj=[[0.03991315450844621, 0.9930167610864072, 0.9135003615055642, 0.009120538516093886, -1.2458519914917328, 0.31047455775216337], [0.040055321748227214, 0.9799054880196877, 0.9201158535321914, 0.009098297454395712, -1.2655792300051798, 0.3105143963048488], [0.04019862987977225, 0.9678113613055614, 0.9287017090887626, 0.009077603688768017, -1.2862596590528592, 0.310561223045458], [0.04034309269627778, 0.9567851149848619, 0.9393424199560593, 0.009058897950976964, -1.3079270348533765, 0.31061531206740023], [0.040488724213308666, 0.9468913255299646, 0.9521505370100193, 0.00904263335053539, -1.330629329570922, 0.3106770954755387], [0.04063553864840286, 0.9382122144379004, 0.9672743554100759, 0.009029287382292596, -1.354432613728079, 0.31074720134062067], [0.041217354769837525, 0.9184952315256895, 1.0526706814325186, 0.009015441282602227, -1.4595468798282256, 0.3111199552645452], [-2.1368931457661615e-09, -0.1299473594859281, -0.505646051548027, -3.4924918731658004e-08, -0.8293562471125323, 2.8154975171483347e-08]]
  do_test(q_traj, 5, 0.25)

