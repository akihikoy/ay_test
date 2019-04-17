float64[] joint_angles

# Poses (x,y,z,qx,qy,qz,qw) of links; [7]*(12+JointNum+1)
float64[] link_x

# Force/torque (fx,fy,fz,tx,ty,tz) of sensors and joints; [6]*(8+JointNum)
float64[] forces

# Masses of links; [12+JointNum+1]
float64[] masses

# Pose (x,y,z,qx,qy,qz,qw) of box1; [7]
float64[] box1_x

# Pose (x,y,z,qx,qy,qz,qw) of links (base, seat-1, seat-2) of chair1; [7]*3
float64[] chair1_x

# Simulation time
float64 time
