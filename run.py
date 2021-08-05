import tensorflow as tf
import scipy.misc
import model
import cv2
from subprocess import call

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "save/model.ckpt")

img = cv2.imread('steering_wheel_image.jpg',0)
rows,cols = img.shape

smoothed_angle = 0

cap = cv2.VideoCapture(0)

while(cv2.waitKey(10) != ord('q')):
	ret, frame = cap.read()
	
	image = scipy.misc.imresize(frame, [66, 200]) / 255.0
	degree = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0] * 180 / scipy.pi
	call('clear')
	print("Predicted steering angle: " + str(degree) + " degrees")
	cv2.imshow('frame', frame)


	smoothed_angle += 0.2 * pow(abs((degree - smoothed_angle)), 2.0 / 3.0) * (degree - smoothed_angle) / abs(degree - smoothed_angle)
	M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
	dst = cv2.warpAffine(img,M,(cols,rows))
	cv2.imshow("steering wheel", dst)
	
    
    
    
cap.release()
cv2.destroyAllWindows()