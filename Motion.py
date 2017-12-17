#Import OpenCV for processing of images
#Import Time for window time logging
#Import DateTime for saving time stamp of motion detection events
#Import Pandas for dataframe and CSV creation

import cv2, time, pandas
from datetime import datetime

#reference background frame against which to compare the presence of object/motion
first_frame=None 

#Capture video feed from webcam (0), use video filename here for pre-recorded video
video = cv2.VideoCapture(0)


statusList=[-1, -1] #stores the presence/absence of object in the present frame. -1 for absent and 1 for present
times=[] #stores timestamps of the entry and exit of object
df=pandas.DataFrame(columns=["Start","End"]) #Pandas dataframe for exporting timestamps to CSV file

#the following loop continuously captures and displays the video feed until user prompts an exit by pressing Q
while True:
	#the read function gives two outputs. The check is a boolean function that returns if the video is being read
	check, frame = video.read() 
	status=-1 #initialise status variable. This stores the presence/absence of object in the current frame
	grayImg=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #Grayscale conversion of the frame
	#Gaussian blur to smoothen the image and remove noise. 
	#The touple is the Kernel size and the 0 is the Std Deviation of the blur function
	grayImg=cv2.GaussianBlur(grayImg, (21,21),0) 
	
	if first_frame is None:
		first_frame=grayImg #collect the reference frame as the first video feed frame
		continue
	
	#calculates the absolute difference between current frame and reference frame
	deltaFrame=cv2.absdiff(first_frame,grayImg)	

	#convert image from grayscale to binary. This increases the demarcation between object and background by using a threshold function that 
	#converts everything above threshold to white
	threshFrame=cv2.threshold(deltaFrame, 30, 255, cv2.THRESH_BINARY)[1]
	
	#dilating the threshold removes the sharp edges at the object/background boundary and makes it smooth. 
	#More the iterations, smoother the image. Too smooth and you lose valuable data
	threshFrame=cv2.dilate(threshFrame, None, iterations=2)
	
	#Contour Function
	#The contour function helps identify the closed object areas within the background. 
	#After thresholding, the frame has closed shapes of the objects against the background
	#The contour function identifies and creates a list (cnts) of all these contours in the frame
	#The RETR_EXTERNAL ensures that you only get the outermost contour details and all child contours inside it are ignored
	#The CHAIN_APPROX_SIMPLE is the approximation method used for locating the contours. The simple one is used here for our trivial purpose
	#Simple approximation removes all the redundant points in the description of the contour line	
	(_,cnts,_)=cv2.findContours(threshFrame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	for contour in cnts:
		if cv2.contourArea(contour) < 10000: 
			#excluding too small contours. Set 10000 (100x100 pixels) for objects close to camera
			continue
		status=1
		#obtain the corresponding bounding rectangle of our detected contour
		(x, y, w, h) = cv2.boundingRect(contour)
		
		#superimpose a rectangle on the identified contour in our original colour image
		#(x,y) is the top left corner, (x+w, y+h) is the bottom right corner
		#(0,255,0) is colour green and 3 is the thickness of the rectangle edges
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
		
		#do the above for all contours greater than our set size
	
	#add the present status to our list
	statusList.append(status)
	
	#Detecting the entry and exit of objects
	#Every entry/exit is identified by a sign change of the last two elements in our list, hence product is -1
	if (statusList[-1]*statusList[-2])==-1:
		times.append(datetime.now())
		
	#unitTesting
	#cv2.imshow("Capturing", grayImg) 
	#cv2.imshow("DeltaFrame", deltaFrame)
	#cv2.imshow("Threshold Frame", threshFrame)
	#print(status)
	
	#displays the continous feed with the green frame for any foreign object in frame
	cv2.imshow("Colour Frame", frame)
	
	#picks up the key press Q and exits when pressed
	key=cv2.waitKey(1)
	if key==ord('q'):
		#if foreign object is in frame at the time of exiting, it stores the timestamp
		if status==1:
			times.append(datetime.now())
		break


#print(statusList)
#print(times)

#take every 2 timestamps in the list and store them as startTime and endTime in the Pandas dataframe
for i in range(0, len(times), 2):
	df=df.append({"Start":times[i], "End":times[i+1]}, ignore_index=True)
	
#Export to csv
df.to_csv("Times.csv")

#Closes all windows
cv2.destroyAllWindows()

#Releases video file/webcam
video.release()


