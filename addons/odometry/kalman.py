import numpy as np


class Kalman:
    
    def __init__(self, f = np.eye(3), h = np.diag([1.,1., 1.]),
                       q = 2*np.diag([1,2,3]), r = 5*np.eye(3)):

        self.I = np.eye(4)
        self.Q = q # Q - ковариационная матрица ошибки модели
        self.R = r # R - ковариационная матрица ошибки измерения
        self.F = f # F - матрица процесса - размер dim_x на dim_x  (3х3)
        self.H = h # H - матрица наблюдения - dim_z на dim_x  (1x3)
        self.dt = 0.1 # H - матрица наблюдения - dim_z на dim_x  (1x3)
    
    def set_state(self):
        self.state =  np.array([[0], [0], [0.05*np.pi], [2.2]]) #self.model.state # start position (x, y, psi, v)
        self.covariance = np.eye(len(self.state))*100.0
    
    def predict(self):
        # time update - prediction
        self.state[0] = self.state[0] + self.dt*self.state[3]*np.cos(self.state[2])
        self.state[1] = self.state[1] + self.dt*self.state[3]*np.sin(self.state[2])
        self.state[2] = (self.state[2]+ np.pi) % (2.0*np.pi) - np.pi
        self.state[3] = self.state[3]
        
        a13 = float(-self.dt*self.state[3]*np.sin(self.state[2]))
        a14 = float(self.dt*np.cos(self.state[2]))
        a23 = float(self.dt*self.state[3]*np.cos(self.state[2]))
        a24 = float(self.dt*np.sin(self.state[2]))
        self.F = np.array([[1.0, 0.0, a13, a14],
                        [0.0, 1.0, a23, a24],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]])
        self.covariance = self.F@self.covariance@self.F + self.Q
    
    def update(self, data):
        # measurement update - correction
        # Measurement Function
        hx = np.array([[float(self.state[0])],
                       [float(self.state[1])]])
        S = self.H@self.covariance@self.H.T + self.R
        K = self.covariance@self.H.T @ np.linalg.inv(S)
        
        self.state = self.state + K@(data - hx)
        self.covariance = (self.I - K@self.H)@self.covariance