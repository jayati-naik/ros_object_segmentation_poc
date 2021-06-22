# ros_object_segmentation_poc

# How to run
1. Add the content of this repo in ~/<catkin_ws>/src/object_segmentation_poc
2. Run catkin_make
3. Start the Depth camera
    ''' roslaunch realsense2_camera rs_camera.launch filters:=pointcloud '''
5. launch the launcher
    ''' roslaunch obj_segmentation_poc obj_segmentation_poc.launch '''
