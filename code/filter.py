import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg
import numpy.random as random
from numpy import dot



def multiply_gaussians(mu1, std1, mu2, std2):
    std1 = std1 if std1!=0 else 1.e-80
    std2 = std2 if std2!=0 else 1.e-80
    mean = (mu2*std1**2 + mu1*std2**2) / (std1**2+std2**2)
    std  = np.sqrt(1. / (1./(std1**2) + 1./(std2**2)))
    return (mean, std)

def add_gaussians(mu1, std1, mu2, std2):
    mean = mu1 + mu2
    std  = np.sqrt(std1**2 + std2**2)
    return mean, std


def g_h_filter(measurements, init_pos, init_vel, g, h):
    print "using g_h_filter at filter.py"
    fltr_pos      = init_pos
    fltr_vel      = init_vel
    fltr_pos_list = [init_pos]

    for meas_pos in measurements[1:]:
        # predict
        pred_pos = fltr_pos + fltr_vel

        # update
        residual = meas_pos - pred_pos
        fltr_pos = pred_pos + g * residual
        fltr_vel = fltr_vel + h * residual
    
        fltr_pos_list.append(fltr_pos) 
    return fltr_pos_list

def plot_moves(real_x, measured_x=None, filter_x=None, g=None, h=None, ylim=None):
    time = np.arange(len(real_x))
    plt.plot(time, real_x, lw=2, color="red", alpha=0.5, label="real")    
    if measured_x!=None:
        plt.plot(time, measured_x, lw=1, color="black", ls="--", label="measured", alpha=.5)
    if filter_x!=None:
        plt.plot(time, filter_x, lw=2, color="green", alpha=0.8, label="filter")
        
    if g!=None and h!=None:
        plt.title("$g=%.2f$    $h=%.2f$"%(g,h))
    plt.legend(loc='lower right') #, bbox_to_anchor=(1, 0.5))
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])

def plot_moves_variance(real, measures, xhist, Phist, ylim=None):
    x = np.array(xhist)
    p = np.array(Phist)
    if real!=None:
        plt.plot(range(len(real)), real, label="real", color="red", lw=10, alpha=0.5)
    if measures!=None:
        plt.plot(range(len(measures)), measures, label="measure", ls=":", color="black")
    plt.plot(range(len(x)), x, label="kalman", color="black", alpha=0.7, lw=2)
    plt.plot(range(len(x)), x+p, color="gray", alpha=0.2)
    plt.plot(range(len(x)), x-p, color="gray", alpha=0.2)
    plt.fill_between(range(len(x)), x+p, x-p, color="yellow", alpha=0.8)
    plt.legend(loc='lower right') #, bbox_to_anchor=(1, 0.5))
    if ylim is None:
        plt.ylim(np.min(measures), np.max(measures))
    else:
        plt.ylim(ylim[0], ylim[1])


def plot_kalman_gh(pos, mpos, init_pos, init_pos_std, init_vel, init_vel_std, g, h, measurement_noise):

    kf = KalmanFilter1D(init_pos, init_pos_std, measurement_noise, init_vel_std)

    f_pos_mu_history, f_pos_std_history = kf.filter_data(mpos, init_vel)    
    
    fig = plt.figure(figsize=(13,4))
    fig.add_subplot(121)
    plot_moves_variance(pos, mpos, f_pos_mu_history, f_pos_std_history)
    plt.ylim(-np.max(mpos), np.max(mpos))

    fig.add_subplot(122)
    g,h = .3,.2
    filter_positions = g_h_filter(mpos, init_pos=init_pos, init_vel=init_vel, g=g, h=h)
    plot_moves(pos, mpos, filter_positions, g, h)
    plt.ylim(-np.max(mpos), np.max(mpos))

class ConstantVelocityRobot(object):
    def __init__(self, init_pos=0., vel=1., mnoise=0.):
        self.pos = init_pos
        self.vel = vel
        self.mnoise = mnoise
        self.pos_history = [init_pos]
        self.m_history   = [self.measure_position()]
    
    def measure_position(self):
        return self.pos + np.random.normal(0, self.mnoise) if self.mnoise!=0 else self.pos
        
    def move(self):
        self.pos += self.vel
        self.pos_history.append(self.pos)
        self.m_history.append(self.measure_position())
    
    def move_n_time_steps(self, n):
        for t in range(n):
            self.move()


def Q_discrete_white_noise(dim, dt=1., var=1.):
    assert dim == 2 or dim == 3
    if dim == 2:
        Q = np.array([[.25*dt**4, .5*dt**3],
                   [ .5*dt**3,    dt**2]], dtype=float)
    else:
        Q = np.array([[.25*dt**4, .5*dt**3, .5*dt**2],
                   [ .5*dt**3,    dt**2,       dt],
                   [ .5*dt**2,       dt,        1]], dtype=float)

    return Q * var


def plot_kalman_position_velocity(phist, mhist, xhist, Phist, vhist=None, ylim=None):
  fig = plt.figure(figsize=(13,4))
  fig.add_subplot(121)
  plot_moves_variance(real=phist, measures=mhist, xhist=[i[0] for i in xhist], Phist=[i[0,0] for i in Phist], ylim=ylim)
  plt.title("position")
  fig.add_subplot(122)
  velocities = [i[1] for i in xhist]
  vmin, vmax = np.min(velocities), np.max(velocities)
  plot_moves_variance(real=vhist, measures=None, xhist=velocities,  Phist=[i[1,1] for i in Phist], ylim = (vmin-np.abs(vmin)*0.2, vmax+np.abs(vmax)*0.2))
  plt.title("velocity")

from math import *
from random import gauss
import numpy as np
import bisect

class RobotPosition:
    def __init__(self, landmarks, world_size):
        self.x = random.random() * world_size
        self.y = random.random() * world_size
        self.orientation = random.random() * 2.0 * pi
        self.forward_noise = 0.0;
        self.turn_noise    = 0.0;
        self.sense_noise   = 1.0;
        self.landmarks  = landmarks
        self.world_size = world_size
    
    def set(self, new_x, new_y, new_orientation):
        if new_x < 0 or new_x >= self.world_size:
            raise ValueError, 'X coordinate out of bound'
        if new_y < 0 or new_y >= self.world_size:
            raise ValueError, 'Y coordinate out of bound'
        if new_orientation < 0 or new_orientation >= 2 * pi:
            raise ValueError, 'Orientation must be in [0..2pi]'
        self.x = float(new_x)
        self.y = float(new_y)
        self.orientation = float(new_orientation)
    
    
    def set_noise(self, new_f_noise, new_t_noise, new_s_noise):
        # makes it possible to change the noise parameters
        # this is often useful in particle filters
        self.forward_noise = float(new_f_noise);
        self.turn_noise    = float(new_t_noise);
        self.sense_noise   = float(new_s_noise);
    
    
    def sense(self):
        Z = []
        for i in range(len(self.landmarks)):
            dist = sqrt((self.x - self.landmarks[i][0]) ** 2 + (self.y - self.landmarks[i][1]) ** 2)
            dist += gauss(0.0, self.sense_noise)
            Z.append(dist)
        return Z
    
    
    def move(self, turn, forward):
        if forward < 0:
            raise ValueError, 'Robot cant move backwards'         
        
        # turn, and add randomness to the turning command
        orientation = (self.orientation + float(turn) + gauss(0.0, self.turn_noise)) % (2*pi)
        
        # move, and add randomness to the motion command
        dist = float(forward) + gauss(0.0, self.forward_noise)
        x = ( self.x + (cos(orientation) * dist) ) % self.world_size
        y = ( self.y + (sin(orientation) * dist) ) % self.world_size
        
        # set particle
        res = RobotPosition(self.landmarks, self.world_size)
        res.set(x, y, orientation)
        res.set_noise(self.forward_noise, self.turn_noise, self.sense_noise)
        res.landmarks = self.landmarks
        res.world_size = self.world_size
        return res
    
    def __repr__(self):
        return '[x=%.6s y=%.6s orient=%.6s]' % (str(self.x), str(self.y), str(self.orientation))
    
class KalmanFilter:

    def __init__(self, dim_x, dim_z, P=0., R=0., Q=None, F=0, H=0, init_x=None, dim_u=0):
        self.x = np.zeros((dim_x, 1)) if init_x is None else init_x # state
        self.P = np.eye(dim_x)*P if np.isscalar(P) else P           # uncertainty covariance
        self.Q = np.eye(dim_x)*Q if np.isscalar(Q) else Q           # process noise
        self.u = np.zeros((dim_u, 1))                               # motion vector
        self.B = 0                           # control transition matrix
        self.F = F                           # state transition matrix
        self.H = H                           # Measurement function
        self.R = np.eye(dim_z)*R             # state uncertainty

        # identity matrix. Do not alter this. 
        self._I = np.eye(dim_x)


    def update(self, Z):

        if Z is None:
            return

        # error (residual) between measurement and prediction
        y = Z - dot(self.H, self.x)

        # project system uncertainty into measurement space 
        S = dot(self.H, self.P).dot(self.H.T) + self.R

        # map system uncertainty into kalman gain
        K = dot(self.P, self.H.T).dot(linalg.inv(S))

        # predict new x with residual scaled by the kalman gain
        self.x += dot(K, y)

        I_KH = self._I - dot(K, self.H)
        self.P = I_KH.dot(self.P) #.dot(I_KH.T) + dot(K, self.R).dot(K.T)


    def predict(self, u=0):
        
        self.x = dot(self.F, self.x) + dot(self.B, u)
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q
        
    def filter_data(self, data):
        count = len(data)
        xhist = [None] * count
        Phist = [None] * count
        
        for t in range(count):            
            z = data[t]
            xhist[t] = np.copy(self.x)
            Phist[t] = np.copy(self.P)
            
            # perform the kalman filter steps
            self.update(z)
            self.predict()

        return xhist, Phist
    

def eval(r, p, world_size):
    if len(p)==0:
        return [],[],[]
    
    errs = []
    for i in range(len(p)): # calculate mean error
        dx = (p[i].x - r.x + (world_size/2.0)) % world_size - (world_size/2.0)
        dy = (p[i].y - r.y + (world_size/2.0)) % world_size - (world_size/2.0)
        err = sqrt(dx * dx + dy * dy)
        errs.append(err)
        
    sx, sy = np.std(np.array([[i.x,i.y] for i in p]), axis=0)

    return np.mean(errs), np.std(errs), np.mean([sx,sy])
    
    
def plot_world(world_size, landmarks, robot_position=None, particles=None, show_pct=0.5, no_title=False):
    plt.ylim(0,world_size)
    plt.xlim(0,world_size)
    
    for l in landmarks:
        plt.scatter(l[0],l[1], c="blue", alpha=0.5, s=200)

    title = ""
    if robot_position is not None:
        title = "$x=%.3f$ $y=%.3f$ "%(robot_position.x, robot_position.y)
        
    if particles is not None and len(particles)!=0:
        avgx, avgy = 0,0
        for i in range(len(particles)):
            avgx, avgy = avgx + particles[i].x, avgy + particles[i].y
            if np.random.random()<show_pct:
                plt.scatter(particles[i].x, particles[i].y, c="black", s=10, alpha=0.2)
        avgx, avgy = avgx / len(particles), avgy/len(particles)
        err = np.sqrt( (avgx-robot_position.x)**2 + (avgy-robot_position.y)**2 )
        err = eval(robot_position, particles, world_size)
        title += "particles $\overline{x}=%.1f$ $\overline{y}=%.1f$, $e=%.2f$ $\pm%.2f$ "%(avgx,avgy,err[0], err[1])
        
        mx,my = np.mean(np.array([[p.x,p.y] for p in particles]), axis=0)
        plt.scatter(mx,my, marker="x", linewidth=3, s=100, color="green")
    
        
    if not no_title:
        plt.title(title)

    if robot_position!=None:
        plt.scatter(robot_position.x,robot_position.y, marker="x", linewidth=3, s=100, color="red")

        
        
def plot_all(world_size, landmarks, robot_positions, particles, errs_mean, errs_std, parts_std):
    fig=plt.figure(figsize=(14,5))
    fig.add_subplot(121)
    plot_world(wsize, lmarks, robot_positions[-1], particles, show_pct=0.2)
    plt.scatter([i.x for i in robot_positions], [i.y for i in robot_positions], color="green", s=10, alpha=0.5)
    if len(particles)>0:
        fig.add_subplot(122)
        errs_mean = np.array(errs_mean)
        errs_std  = np.array(errs_std)
        plt.plot(range(len(errs_mean)), errs_mean, label="err to pos", ls":")
        plt.fill_between(range(len(errs_mean)), errs_mean-errs_std, errs_mean+errs_std, color="yellow", alpha=0.5)
        plt.plot(range(len(parts_std)), parts_std, label="parts std")
        plt.xlabel("step")
        plt.ylabel("meter")
        plt.legend()
