# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 14:44:04 2019

@author: Kitsune
"""

from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2
from matplotlib import pyplot as plt

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

	
	
img = cv2.imread('test2edit.jpg')

minthresh = np.array([95,95,50],np.uint8)
maxthresh = np.array([255,255,255],np.uint8)
testimg = cv2.cvtColor(img,cv2.cv2.COLOR_BGR2RGB)
testthreshold = cv2.inRange(img,minthresh,maxthresh)
cv2.imshow("testthreshold.png",testthreshold)
# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(testthreshold,cv2.MORPH_OPEN,kernel, iterations = 2)
cv2.imshow('testthresh2.png',opening)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#cv2.imshow('gray',gray)
#ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('image',thresh)

ret, thresh2 = cv2.threshold(gray,245,255,cv2.THRESH_BINARY)
cv2.imshow('image2',thresh2)

#kernel = np.ones((5,5),np.uint8)
#erosion = cv2.erode(thresh,kernel,iterations = 1)
#cv2.imshow('image2',erosion)

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
opening2 = cv2.morphologyEx(thresh2,cv2.MORPH_OPEN,kernel, iterations = 2)
cv2.imwrite('chalky.png',opening2)
# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)


# find contours in the thresholded image
cnts = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
(cnts, _) = contours.sort_contours(cnts)
print("[INFO] {} unique contours found".format(len(cnts)))
lengths = np.array([-1,-1.0])
# loop over the contours
for (i, c) in enumerate(cnts):
	#compute the rotated bounding box of the contour
	orig = img.copy()
	box = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	box = np.array(box, dtype="int")
    
	# unpack the ordered bounding box, then compute the midpoint
    # between the top-left and top-right coordinates, followed by
	# the midpoint between bottom-left and bottom-right coordinates
	(tl, tr, br, bl) = box
	(tltrX, tltrY) = midpoint(tl, tr)
	(blbrX, blbrY) = midpoint(bl, br)
 
	# compute the midpoint between the top-left and top-right points,
	# followed by the midpoint between the top-righ and bottom-right
	(tlblX, tlblY) = midpoint(tl, bl)
	(trbrX, trbrY) = midpoint(tr, br)
 
	# draw the midpoints on the image
	cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
 
	# draw lines between the midpoints
	cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
		(255, 0, 255), 2)
	cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
		(255, 0, 255), 2)
    
	# compute the Euclidean distance between the midpoints
	dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
	dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
    
    # compute the Euclidean distance between the midpoints
	print(dA)
	if dA>dB:
		#lengths=np.append(lengths,[i,dA])
		lengths = np.vstack((lengths,[i,dA]))
	else:
		#lengths=np.append(lengths,[i,dB])
		lengths = np.vstack((lengths,[i,dB]))


    # draw the object sizes on the image
	cv2.putText(orig, "{:.1f}px".format(dA),
		(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)
	cv2.putText(orig, "{:.1f}px".format(dB),
		(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)
    # draw the contour
	print("for grain %i da is %i db is %i"%((i+1),dA,dB))
    # draw the contour
	((x, y), _) = cv2.minEnclosingCircle(c)
	cv2.putText(img, "#{}".format(i + 1), (int(x) - 10, int(y)),
		cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
	cv2.drawContours(img, [c], -1, (0, 255, 0), 2)


print(lengths)

lengths=lengths[lengths[:,1].argsort()]
lengths = lengths[::-1]
print("new")
print(lengths)
print("only lengths")
print(lengths[:,1])
sum=0
numberofgood=0
average=0
for(i, y) in enumerate(lengths[:,1]):
	print("y for %i is %f" %(i,y))
	limit=average*0.75
	print("75percent average is %f" %(limit))
	if limit>=y:
		break
	else :
		sum=sum+y
		average=sum/(i+1)
		numberofgood=i

		
print("number of good %i" %(numberofgood))
print("broken are")
broken=lengths[numberofgood:,0]
print(lengths[numberofgood-2:,0])


# show the output image
cv2.imshow("Image", img)
cv2.namedWindow('test',cv2.WINDOW_NORMAL)
cv2.resizeWindow('test', 1000,1000)
cv2.imshow("test",orig)

cv2.imwrite('count.png',orig)
for(i, c) in enumerate(cnts):
	if (i in broken):
		((x, y), _) = cv2.minEnclosingCircle(c)
		cv2.putText(img, "#{}".format(i + 1), (int(x) - 10, int(y)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
		cv2.drawContours(img, [c], -1, (255, 0, 0), 2)


cv2.namedWindow('broken',cv2.WINDOW_NORMAL)
cv2.resizeWindow('broken', 1000,1000)
cv2.imshow("broken",img)		
cnts = cv2.findContours(opening2.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
print("[INFO] {} unique contours found".format(len(cnts)))
 
# loop over the contours
for (i, c) in enumerate(cnts):
	# draw the contour
	((x, y), _) = cv2.minEnclosingCircle(c)
	cv2.putText(img, "#{}".format(i + 1), (int(x) - 10, int(y)),
		cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
	cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
    
cv2.imshow("chalky", img)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
