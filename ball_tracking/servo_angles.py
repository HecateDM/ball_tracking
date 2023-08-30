#!/usr/bin/env python
#todo script é um nó diferente
#esse nó pega as coordenadas da bola que foram publicadas pelo nó "ball_tracking" e publica os angulos para o servo
import rospy
from std_msgs.msg import Float32MultiArray, Int16MultiArray

import yaml
import numpy as np
import math as mt


pub_angles = rospy.Publisher('/servo', Int16MultiArray,queue_size=1)
angles = Int16MultiArray()
import time

with open('/home/dimitria/ball_tracking/ball_tracking/params2.yaml') as f:
    cam_params = yaml.load(f, Loader=yaml.FullLoader)
    intrinsic = np.array(cam_params['mtx'])

fx = intrinsic[0,0]
fy = intrinsic[1,1]

cx = 320
cy = 240


global theta_z, theta_y
theta_z = 0   #< >
theta_y = -50 #cima e baixo ^v


#time.sleep(5)

def angles_callback(msg):#toda vez wue eu receber a posicao da bola eu vou executar o callback
    #esse calback calcula os 2 angulos da cabeça e publica
    global theta_y, theta_z
    print
    if msg.data[0] !=1000 and msg.data[1] !=1000:
        v = msg.data[0]
        u = msg.data[1]
                                                
        if abs(v - cx) > 25 or abs(u - cy >20):
            x = -(v-cx)
            y = -(u-cy)
            #if (x >=15) and (y>=15):
            theta_z = theta_z + int(np.arctan2(x,fx)*180/mt.pi) 
            theta_y = theta_y + int(np.arctan2(y,fy)*180/mt.pi) 
            if(theta_y<-70):
                theta_y = -70
            if theta_y > -20:
                theta_y = -20
            if theta_z < -50:
                theta_z = -50 
            if theta_z > 50:
                theta_z = 50

            angles.data = [theta_z,theta_y]
        
    else:
        theta_z = 0
        theta_y = -50
        angles.data = [theta_z,theta_y]
        
    rospy.loginfo('the angles are  %i,%i', theta_y,theta_z)
    pub_angles.publish(angles)

    #time.sleep(0.8)

def angles_sub():
    rospy.Subscriber('/ball_pose', Float32MultiArray,angles_callback)
    rospy.spin()

if __name__=='__main__':
    rospy.init_node('angles_publisher')
    rospy.loginfo('angles_pub node started')
    angles.data = [0,-50]
    pub_angles.publish(angles)
    angles_sub()