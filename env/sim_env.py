import numpy as np
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.transforms as transforms
import matplotlib.cm as cm
import matplotlib.image as mpimg
from gymnasium import spaces
from math_tool import *
import matplotlib.backends.backend_agg as agg
from PIL import Image
import random
import copy

class UAVEnv:
    def __init__(self,length=2,num_obstacle=3,num_agents=4):
        self.length = length # side length of cubic boundary [0, length]^3
        self.num_obstacle = num_obstacle # number of obstacles
        self.num_agents = num_agents
        self.time_step = 0.5 # update time step
        self.v_max = 0.1
        self.v_max_e = 0.12
        self.a_max = 0.04
        self.a_max_e = 0.05
        self.L_sensor = 0.2
        self.num_lasers = 32 # num of laser beams (3D needs more rays than 2D's 16)
        self.multi_current_lasers = [[self.L_sensor for _ in range(self.num_lasers)] for _ in range(self.num_agents)]
        self.agents = ['agent_0','agent_1','agent_2','target']
        self.info = np.random.get_state() # get seed
        self.obstacles = [obstacle() for _ in range(self.num_obstacle)]
        self.history_positions = [[] for _ in range(num_agents)]

        # 3D actions: [a_x, a_y, a_z]
        self.action_space = {
            'agent_0': spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
            'agent_1': spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
            'agent_2': spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
            'target': spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
            } # action represents [a_x, a_y, a_z]

        # Observation dimensions for 3D:
        # Hunter UAV (agent_0,1,2): S_uavi(6) + S_team(6) + S_target(3) + S_obser(32) = 47
        #   S_uavi: [px/L, py/L, pz/L, vx/v_max, vy/v_max, vz/v_max]
        #   S_team: 2 other hunters x 3d pos = 6
        #   S_target: [d/(sqrt(3)L), theta_azimuth, phi_polar] = 3
        #   S_obser: num_lasers (32)
        # Target UAV: S_uavi(6) + S_obser(32) + S_evade_d(3 distances) = 41
        self.observation_space = {
            'agent_0': spaces.Box(low=-np.inf, high=np.inf, shape=(47,)),
            'agent_1': spaces.Box(low=-np.inf, high=np.inf, shape=(47,)),
            'agent_2': spaces.Box(low=-np.inf, high=np.inf, shape=(47,)),
            'target': spaces.Box(low=-np.inf, high=np.inf, shape=(41,))
        }
        

    def reset(self):
        SEED = random.randint(1,1000)
        random.seed(SEED)
        self.multi_current_pos = []
        self.multi_current_vel = []
        self.history_positions = [[] for _ in range(self.num_agents)]
        for i in range(self.num_agents):
            if i != self.num_agents - 1: # if not target
                self.multi_current_pos.append(np.random.uniform(low=0.1,high=0.4,size=(3,)))
            else: # for target
                self.multi_current_pos.append(np.random.uniform(low=0.1,high=0.4,size=(3,)))
            self.multi_current_vel.append(np.zeros(3)) # initial velocity = [0,0,0]

        # update lasers
        self.update_lasers_isCollied_wrapper()
        ## multi_obs is list of agent_obs, state is multi_obs after flattenned
        multi_obs = self.get_multi_obs()
        return multi_obs

    def step(self,actions):
        last_d2target = []
        for i in range(self.num_agents - 1):

            pos = self.multi_current_pos[i]
            if i != self.num_agents - 1:
                pos_taget = self.multi_current_pos[-1]
                last_d2target.append(np.linalg.norm(pos-pos_taget))
            
            # 3D velocity update
            self.multi_current_vel[i][0] += actions[i][0] * self.time_step
            self.multi_current_vel[i][1] += actions[i][1] * self.time_step
            self.multi_current_vel[i][2] += actions[i][2] * self.time_step
            vel_magnitude = np.linalg.norm(self.multi_current_vel[i])
            if i != self.num_agents - 1:
                if vel_magnitude >= self.v_max:
                    self.multi_current_vel[i] = self.multi_current_vel[i] / vel_magnitude * self.v_max
            else:
                if vel_magnitude >= self.v_max_e:
                    self.multi_current_vel[i] = self.multi_current_vel[i] / vel_magnitude * self.v_max_e

            # 3D position update
            self.multi_current_pos[i][0] += self.multi_current_vel[i][0] * self.time_step
            self.multi_current_pos[i][1] += self.multi_current_vel[i][1] * self.time_step
            self.multi_current_pos[i][2] += self.multi_current_vel[i][2] * self.time_step

        # Update obstacle positions (3D)
        for obs in self.obstacles:
            obs.position += obs.velocity * self.time_step
            # Check for boundary collisions and adjust velocities in 3D
            for dim in [0, 1, 2]:
                if obs.position[dim] - obs.radius < 0:
                    obs.position[dim] = obs.radius
                    obs.velocity[dim] *= -1
                elif obs.position[dim] + obs.radius > self.length:
                    obs.position[dim] = self.length - obs.radius
                    obs.velocity[dim] *= -1

        Collided = self.update_lasers_isCollied_wrapper()
        rewards, dones= self.cal_rewards_dones(Collided,last_d2target)   
        multi_next_obs = self.get_multi_obs()
        # sequence above can't be disrupted

        return multi_next_obs, rewards, dones, Collided

    def test_multi_obs(self):
        total_obs = []
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]
            S_uavi = [
                pos[0]/self.length,
                pos[1]/self.length,
                pos[2]/self.length,
                vel[0]/self.v_max,
                vel[1]/self.v_max,
                vel[2]/self.v_max
            ]
            total_obs.append(S_uavi)
        return total_obs
    
    def get_multi_obs(self):
        total_obs = []
        single_obs = []
        S_evade_d = [] # dim 3 only for target (3 hunter distances)
        diag_L = np.sqrt(3) * self.length  # spatial diagonal of cube for normalization
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]
            S_uavi = [
                pos[0]/self.length,
                pos[1]/self.length,
                pos[2]/self.length,
                vel[0]/self.v_max,
                vel[1]/self.v_max,
                vel[2]/self.v_max
            ] # dim 6
            S_team = [] # dim 3*2 = 6 (2 other hunters, 3D pos)
            S_target = [] # dim 3 (distance + azimuth + polar)
            for j in range(self.num_agents):
                if j != i and j != self.num_agents - 1: 
                    pos_other = self.multi_current_pos[j]
                    S_team.extend([pos_other[0]/self.length, pos_other[1]/self.length, pos_other[2]/self.length])
                elif j == self.num_agents - 1:
                    pos_target = self.multi_current_pos[j]
                    rel = pos_target - pos
                    d = np.linalg.norm(rel)
                    # azimuth angle in xy plane (theta)
                    theta = np.arctan2(rel[1], rel[0])
                    # polar angle from z-axis (phi), range [0, pi]
                    phi = np.arccos(np.clip(rel[2] / (d + 1e-9), -1, 1))
                    S_target.extend([d/diag_L, theta / np.pi, phi / np.pi])
                    if i != self.num_agents - 1:
                        S_evade_d.append(d/diag_L)

            S_obser = self.multi_current_lasers[i] # dim num_lasers (32)

            if i != self.num_agents - 1:
                single_obs = [S_uavi, S_team, S_obser, S_target]
            else:
                single_obs = [S_uavi, S_obser, S_evade_d]
            _single_obs = list(itertools.chain(*single_obs))
            total_obs.append(_single_obs)
            
        return total_obs

    def cal_rewards_dones(self,IsCollied,last_d):
        dones = [False] * self.num_agents
        rewards = np.zeros(self.num_agents)
        mu1 = 0.9 # r_near
        mu2 = 0.2 # r_safe
        mu3 = 0.0 # r_multi_stage (disabled by default in original)
        mu4 = 5 # r_finish
        d_capture = 0.3
        d_limit = 0.75
        ## 1 reward for single rounding-up-UAVs:
        for i in range(3):
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]
            pos_target = self.multi_current_pos[-1]
            v_i = np.linalg.norm(vel)
            dire_vec = pos_target - pos
            d = np.linalg.norm(dire_vec) # distance to target

            cos_v_d = np.dot(vel,dire_vec)/(v_i*d + 1e-3)
            r_near = abs(2*v_i/self.v_max)*cos_v_d
            rewards[i] += mu1 * r_near
        
        ## 2 collision reward for all UAVs:
        for i in range(self.num_agents):
            if IsCollied[i]:
                r_safe = -10
            else:
                lasers = self.multi_current_lasers[i]
                r_safe = (min(lasers) - self.L_sensor - 0.1)/self.L_sensor
            rewards[i] += mu2 * r_safe

        ## 3 multi-stage's reward for rounding-up-UAVs (3D adapted)
        p0 = self.multi_current_pos[0]
        p1 = self.multi_current_pos[1]
        p2 = self.multi_current_pos[2]
        pe = self.multi_current_pos[-1]
        # Use tetrahedron volumes for 3D stage detection
        V1 = cal_tetrahedron_V(p0, p1, p2, pe)  # volume of hunters-target tetra
        V_hunters = cal_tetrahedron_V(p0, p1, p2, p0)  # always 0, placeholder
        # Use 3 pairwise face areas (triangles with target) to approximate "Sum_S"
        S1 = cal_triangle_S(p0, p1, pe)
        S2 = cal_triangle_S(p1, p2, pe)
        S3 = cal_triangle_S(p2, p0, pe)
        S4 = cal_triangle_S(p0, p1, p2)
        d1 = np.linalg.norm(p0 - pe)
        d2 = np.linalg.norm(p1 - pe)
        d3 = np.linalg.norm(p2 - pe)
        Sum_S = S1 + S2 + S3
        Sum_d = d1 + d2 + d3
        Sum_last_d = sum(last_d)

        # Check 3D encirclement
        is_encircled, avg_d, all_in_range = check_encirclement_3d([p0, p1, p2], pe, d_capture)

        # 3.1 reward for target UAV (evasion):
        rewards[-1] += np.clip(2 * (Sum_d - Sum_last_d), -2, 2)
        # 3.2 stage-1 track (far, approaching)
        if Sum_S > S4 and Sum_d >= d_limit and all(d >= d_capture for d in [d1, d2, d3]):
            r_track = - Sum_d / max([d1, d2, d3])
            rewards[0:3] += mu3 * r_track
        # 3.3 stage-2 encircle (close, forming enclosure)
        elif Sum_S > S4 and (Sum_d < d_limit or any(d <= d_capture for d in [d1, d2, d3])):
            r_encircle = -1/3 * np.log(Sum_S - S4 + 1 + V1 * 100)  # add volume term
            rewards[0:3] += mu3 * r_encircle
        # 3.4 stage-3 capture (final push)
        elif (not is_encircled) and any(d > d_capture for d in [d1, d2, d3]):
            r_capture = np.exp((Sum_last_d - Sum_d) / (3 * self.v_max))
            rewards[0:3] += mu3 * r_capture
        
        ## 4 finish rewards: 3D encirclement + capture radius satisfied
        if is_encircled:
            print('add 4', mu4 * 10, '  [3D ENCIRCLEMENT + CAPTURE]')
            rewards[0:3] += mu4 * 10
            dones = [True] * self.num_agents
        return rewards, dones

    def update_lasers_isCollied_wrapper(self):
        self.multi_current_lasers = []
        dones = []
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            current_lasers = [self.L_sensor] * self.num_lasers
            done_obs = []
            for obs in self.obstacles:
                obs_pos = obs.position
                r = obs.radius
                _current_lasers, done = update_lasers(pos, obs_pos, r, self.L_sensor, self.num_lasers, self.length)
                current_lasers = [min(l, cl) for l, cl in zip(_current_lasers, current_lasers)]
                done_obs.append(done)
            done = any(done_obs)
            if done:
                self.multi_current_vel[i] = np.zeros(3)
            self.multi_current_lasers.append(current_lasers)
            dones.append(done)
        return dones

    def render(self):
        fig = plt.gcf()
        plt.clf()
        ax = fig.add_subplot(111, projection='3d')
        
        # plot round-up-UAVs
        for i in range(self.num_agents - 1):
            pos = copy.deepcopy(self.multi_current_pos[i])
            vel = self.multi_current_vel[i]
            self.history_positions[i].append(pos)
            trajectory = np.array(self.history_positions[i])
            # plot trajectory
            if len(trajectory) > 1:
                ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'b-', alpha=0.3)
            # Plot hunter UAV
            ax.scatter(pos[0], pos[1], pos[2], c='b', s=80, marker='o', label='Hunter' if i == 0 else "")

        # plot target
        pos_t = self.multi_current_pos[-1]
        ax.scatter(pos_t[0], pos_t[1], pos_t[2], c='r', s=120, marker='*', label='Target')
        self.history_positions[-1].append(copy.deepcopy(pos_t))
        trajectory = np.array(self.history_positions[-1])
        if len(trajectory) > 1:
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'r-', alpha=0.3)

        # Plot obstacles as spheres (approximate with scatter + size)
        for obstacle in self.obstacles:
            u, v = np.mgrid[0:2*np.pi:12j, 0:np.pi:8j]
            x = obstacle.position[0] + obstacle.radius * np.cos(u) * np.sin(v)
            y = obstacle.position[1] + obstacle.radius * np.sin(u) * np.sin(v)
            z = obstacle.position[2] + obstacle.radius * np.cos(v)
            ax.plot_wireframe(x, y, z, color='gray', alpha=0.3)

        ax.set_xlim(-0.1, self.length + 0.1)
        ax.set_ylim(-0.1, self.length + 0.1)
        ax.set_zlim(-0.1, self.length + 0.1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend(loc='upper left')
        plt.draw()
        
        # Save the current figure to a buffer
        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        
        # Convert buffer to a NumPy array
        image = np.asarray(buf)
        return image

    def render_anime(self, frame_num):
        fig = plt.gcf()
        plt.clf()
        ax = fig.add_subplot(111, projection='3d')

        for i in range(self.num_agents - 1):
            pos = copy.deepcopy(self.multi_current_pos[i])
            vel = self.multi_current_vel[i]
            self.history_positions[i].append(pos)
            
            trajectory = np.array(self.history_positions[i])
            if len(trajectory) > 1:
                for j in range(len(trajectory) - 1):
                    color = cm.viridis(j / len(trajectory))
                    ax.plot(trajectory[j:j+2, 0], trajectory[j:j+2, 1], trajectory[j:j+2, 2],
                            color=color, alpha=0.7)

            ax.scatter(pos[0], pos[1], pos[2], c='b', s=80, marker='o', label='Hunter' if i == 0 else "")

        pos_e = copy.deepcopy(self.multi_current_pos[-1])
        ax.scatter(pos_e[0], pos_e[1], pos_e[2], c='r', s=120, marker='*', label='Target')
        self.history_positions[-1].append(pos_e)
        trajectory = np.array(self.history_positions[-1])
        if len(trajectory) > 1:
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'r-', alpha=0.3)

        for obstacle in self.obstacles:
            u, v = np.mgrid[0:2*np.pi:12j, 0:np.pi:8j]
            x = obstacle.position[0] + obstacle.radius * np.cos(u) * np.sin(v)
            y = obstacle.position[1] + obstacle.radius * np.sin(u) * np.sin(v)
            z = obstacle.position[2] + obstacle.radius * np.cos(v)
            ax.plot_wireframe(x, y, z, color='gray', alpha=0.3)

        ax.set_xlim(-0.1, self.length + 0.1)
        ax.set_ylim(-0.1, self.length + 0.1)
        ax.set_zlim(-0.1, self.length + 0.1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend(loc='upper left')
        plt.draw()

    def close(self):
        plt.close()

    def check_collisions(self):
        collision_status = [False] * self.num_agents

        # Check UAV-UAV collisions
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                distance = np.linalg.norm(self.multi_current_pos[i] - self.multi_current_pos[j])
                if distance < 0.1:
                    collision_status[i] = True
                    collision_status[j] = True

        # Check UAV-obstacle collisions (3D sphere-sphere)
        for i in range(self.num_agents):
            for obs in self.obstacles:
                distance = np.linalg.norm(self.multi_current_pos[i] - obs.position)
                if distance < obs.radius:
                    collision_status[i] = True

        return collision_status


class obstacle():
    def __init__(self, length=2):
        # 3D position
        self.position = np.random.uniform(low=0.45, high=length-0.55, size=(3,))
        # 3D velocity: sample on unit sphere
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.arccos(np.random.uniform(-1, 1))
        speed = 0.04
        self.velocity = np.array([
            speed * np.sin(phi) * np.cos(theta),
            speed * np.sin(phi) * np.sin(theta),
            speed * np.cos(phi)
        ])
        self.radius = np.random.uniform(0.1, 0.15)
