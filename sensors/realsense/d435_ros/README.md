How to
==================
Src: https://github.com/IntelRealSense/realsense-ros/tree/development/realsense2_description

```
$ rosrun xacro xacro --inorder test_d435_camera.urdf.xacro  -o test_d435_camera.urdf use_nominal_extrinsics:=true add_plug:=true use_mesh:=true

$ roslaunch view_d435_model.launch
```

```
> tf_once 'camera_link','camera_color_optical_frame'
[0.0, 0.015, 0.0, -0.5, 0.49999999999755174, -0.5, 0.5000000000024483]
```
