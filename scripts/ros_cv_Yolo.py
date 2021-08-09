#!/usr/bin/env python
#-*-coding: utf-8-*-
from __future__ import print_function

#import roslib
#roslib.load_manifest('beginner_tutorials')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import time
import numpy as np

class image_converter:

  def __init__(self):
    self.image_pub = rospy.Publisher("image_topic_det",Image)
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/camera/color/image_raw",Image,self.callback)
 
  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    net = cv2.dnn.readNet("/home/robot/obj_det/src/beginner_tutorials/scripts/yolov3-tiny.weights", "/home/robot/obj_det/src/beginner_tutorials/scripts/yolov3-tiny.cfg")
    classes = []
    with open("/home/robot/obj_det/src/beginner_tutorials/scripts/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    
    preTime = 0
    curTime = time.time()
    sec = curTime - preTime
    preTime = curTime
    fps = 1/(sec)
    
    height, width, channels = cv_image.shape
    #cv2.imshow("Video",frame)
       
    # Detecting objects
    blob = cv2.dnn.blobFromImage(cv_image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    #정보를 화면에 표시
    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # 좌표
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
            
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.4) 
    #0.5 = confidence threshold o.4=NMS threshold
    
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            score = confidences[i]

        # 경계상자와 클래스 정보 투영
            cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 0, 255), 4)
            cv2.putText(cv_image, label, (x, y - 20), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 1)
            cv2.putText(cv_image, str(score), (x, y - 40), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 1)
            cv2.putText(cv_image, str(fps), (0,40), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 1)

    cv2.imshow("Image window", cv_image)
    cv2.waitKey(3)

    try:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    except CvBridgeError as e:
      print(e)

def main(args):
  print(cv2.__version__)
  rospy.init_node('Detection', anonymous=True)  
  ic = image_converter()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    
    main(sys.argv)
