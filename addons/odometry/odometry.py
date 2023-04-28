import numpy as np 
import cv2

from .kalman import Kalman

#https://github.com/uoip/monoVO-python

def get_R(alpha):
    M = np.array([[np.cos(np.pi*alpha/180), np.sin(np.pi*alpha/180)],
         [-np.sin(np.pi*alpha/180), np.cos(np.pi*alpha/180)]
        ])
    return M

def show_direction(image, t, M):
    line_thickness = 1
    cx, cy = t
    triangle = np.array([[-9, 9], [9, 9], [0, -11]]).T

    triangle_rot = M@triangle
    triangle = triangle_rot.T
    triangle[:,0] += cx
    triangle[:,1] += cy
    points = [[0,1], [0,2], [1,2]]
    for point in points:
        cv2.line(image, (int(triangle[point[0]][0]),int(triangle[point[0]][1])),
                        (int(triangle[point[1]][0]),int(triangle[point[1]][1])),
                 (0, 0, 255),
                 thickness=line_thickness
                )


dt = 0.1
# Q
GPS     = 11.7*8.8*dt**2  # assume 8.8m/s2 as maximum acceleration, forcing the vehicle
Course  = 1.7*dt # assume 0.2rad/s as maximum turn rate for the vehicle
Velocity= 8.8*dt # assume 8.8m/s2 as maximum acceleration, forcing the vehicle
q = np.diag([GPS**2, GPS**2, Course**2, Velocity**2])
# H
h = np.array([[1.0, 0.0, 0.0, 0.0],
				[0.0, 1.0, 0.0, 0.0]])
# R
varGPS = 0.5 # Standard Deviation of GPS Measurement
r = np.diag([varGPS**2.0, varGPS**2.0])
# F
f = np.eye(4)

def mapping(q_in):
	kalman = Kalman(f = f, h = h, q = q, r = r)
	kalman.set_state()
	traj = np.zeros((400,400,3), dtype=np.uint8)
	while True:
		#raw_frame, frame, coords, frame_id
		_, _, coords, frame_id = q_in.get()
		alpha = coords[3]
		Rt = get_R(alpha)
		x, y, z = coords[0], coords[1], coords[2]
		# Kalman
		# kalman.predict()
		# kalman.update(np.array([[float(coords[0])],
		# 				  [float(coords[2])]]))
		# coords = np.array([[float(kalman.state[0])], 
		# 					coords[1],
		# 					[float(kalman.state[1])]])
		# x, y, z = coords[0], coords[1], coords[2]
		draw_x, draw_y = int(x+200), int(y+200)

		z_color = int(z*255/300)
		#cv2.circle(traj, (draw_x,draw_y), 1, (z_color,255-z_color,255), 2)
		cv2.circle(traj, (draw_x,draw_y), 1, (frame_id/1000,255-frame_id/1000,255), 2)
		cv2.rectangle(traj, (10, 20), (600, 60), (0,0,0), -1)
		text = "Coordinates: x={:.2f}m y={:.2f}m z={:.2f}m".format(x,y,z)
		cv2.putText(traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
		show_direction(traj, (draw_x, draw_y), Rt)
		cv2.imshow('Trajectory', traj)
		cv2.waitKey(1)

if __name__ == '__main__':
	mapping()	
