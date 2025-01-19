#!/usr/bin/python3
import copy

traj=[[1],[2],[3]]
print('traj= ',traj)
goal=traj[-1]
goal[0]=100
print('traj= ',traj)

print('----------')

traj=[[1],[2],[3]]
print('traj= ',traj)
goal=copy.deepcopy(traj[-1])  ###copy.deepcopy###
goal[0]=100
print('traj= ',traj)
