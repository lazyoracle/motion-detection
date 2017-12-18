# Motion Detection in Video Feed using OpenCV with Python

The webcam feed is used as a continuous video source. But it works fine with any video file that can be read using OpenCV. Against the reference of the starting frame, the program detects the presence of any foreign objects in the frame and logs the time stamp of the entry/exit of objects in the frame.

The program doesn't seperately log timestamps of the movements of multiple possible objects. However, it does identify the contours of all the foreign objects and as such can be easily extended to log movement of multiple objects.

The timestamps of the events are finally parsed into a Pandas Dataframe and exported to a CSV in datetime format.

Application - Can be used for tracking movement of babies, pets etc. The implementation is pretty lightweight and can be fired up on a Raspberry Pi with B/W CCTV footage to track the presence of people in a room and log entry/exit.

The code is mostly self-explanatory and contains extensive comments wherever necessary.

Non-native dependencies - OpenCV, Pandas

`pip install opencv-python`

`pip install pandas`

Python version - 3.5 and above (Might work for lower versions of Python 3, not tested)
