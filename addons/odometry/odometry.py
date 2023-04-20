import numpy as np 
import cv2

from kalman import Kalman

#https://github.com/uoip/monoVO-python

def get_R(alpha):
	return np.array([[0][0][0]])

def show_direction(image, t, Rt):
    r = 50
    alpha = 50
    if Rt is None:
        return
    R = Rt[:2, :2]
    x1, y1 = r*np.cos(np.pi*alpha/180+np.pi/2), r*np.sin(np.pi*alpha/180+np.pi/2)
    x2, y2 = r*np.cos(np.pi*alpha/180-np.pi/2), -r*np.sin(np.pi*alpha/180-np.pi/2)
    points = np.array([[x1, y1],[x2, y2]])
    r_points = []
    r_points.append(R @ points[0])
    r_points.append(R @ points[1])
    r_points = np.array(r_points)
    r_points[0] += t
    r_points[1] += t
    line_thickness = 2
    cv2.line(image, (int(r_points[0][0]), int(r_points[0][1])), (int(r_points[1][0]), int(r_points[1][1])), 
					(np.random.randint(255), np.random.randint(255), np.random.randint(255)), thickness=line_thickness)


def mapping(q_in):
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
	kalman = Kalman(f = f, h = h, q = q, r = r)
	kalman.set_state()

	traj = np.zeros((600,600,3), dtype=np.uint8)

	while True:
		raw_frame, frame, coords, frame_id = q_in.get()
		#frame = raw_frame.copy()
		alpha = coords[3]
		Rt = get_R(alpha)
		if coords is not None:
			x, y, z = coords[0], coords[1], coords[2]
			# Kalman
			kalman.predict()
			kalman.update(np.array([[float(coords[0])],
							  [float(coords[2])]]))
			coords = np.array([[float(kalman.state[0])], 
								coords[1],
								[float(kalman.state[1])]])
			x, y, z = coords[0], coords[1], coords[2]
		else:
			x, y, z = 0., 0., 0.
		draw_x, draw_y = int(x), int(y)

		z_color = int(z*255/300)
		#cv2.circle(traj, (draw_x,draw_y), 1, (z_color,255-z_color,255), 2)
		cv2.circle(traj, (draw_x,draw_y), 1, (frame_id/1000,255-frame_id/1000,255), 2)
		cv2.rectangle(traj, (10, 20), (600, 60), (0,0,0), -1)
		text = "Coordinates: x=%2fm y=%2fm z=%2fm"%(x,y,z)
		cv2.putText(traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
		#show_direction(traj, coords[:2], Rt)

		cv2.imshow('Road facing camera', frame)
		cv2.imshow('Trajectory', traj)
		cv2.waitKey(1)

	cv2.imwrite('map.png', traj)


if __name__ == '__main__':
	mapping()	
