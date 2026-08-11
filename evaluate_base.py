from model.maddpg_att import MADDPGWithAttention
from model.maddpg import MADDPG
from env.sim_env import UAVEnv
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import warnings
import os
import time
warnings.filterwarnings('ignore')


def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_velocity_magnitude(time_steps, velocities_magnitude):
    plt.figure(figsize=(15, 4))  
    for i in range(len(velocities_magnitude)):
        if i!=3:
            plt.plot(time_steps, velocities_magnitude[i], label=f'UAV {i}')
        else:
            plt.plot(time_steps, velocities_magnitude[i], label='Target')
    plt.xlabel("Time Steps")
    plt.ylabel("Magnitude")
    plt.title("UAV Velocity Magnitude")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_velocity_x(time_steps, velocities_x):
    plt.figure(figsize=(15, 4))  
    for i in range(len(velocities_x)):
        if i!=3:
            plt.plot(time_steps, velocities_x[i], label=f'UAV {i}')
        else:
            plt.plot(time_steps, velocities_x[i], label='Target')
    plt.xlabel("Time Steps")
    plt.ylabel("$vel_x$")
    plt.title("UAV $Vel_x$")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_velocity_y(time_steps, velocities_y):
    plt.figure(figsize=(15, 4))  
    for i in range(len(velocities_y)):
        if i!=3:
            plt.plot(time_steps, velocities_y[i], label=f'UAV {i}')
        else:
            plt.plot(time_steps, velocities_y[i], label='Target')
    plt.xlabel("Time Steps")
    plt.ylabel("$vel_y$")
    plt.title("UAV $Vel_y$")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_velocities(velocities_magnitude, velocities_x, velocities_y, velocities_z):
    time_steps = range(len(velocities_magnitude[0]))
    fig, axs = plt.subplots(4, 1, figsize=(10, 12))

    for i in range(len(velocities_magnitude)):
        if i != 3:
            axs[0].plot(time_steps, velocities_magnitude[i], label=f'UAV {i}')
        else:
            axs[0].plot(time_steps, velocities_magnitude[i], label=f'Target')
    axs[0].set_title('Speed Magnitude vs Time')
    axs[0].set_xlabel('Time Step')
    axs[0].set_ylabel('Speed Magnitude')
    axs[0].legend()

    for i in range(len(velocities_x)):
        if i != 3:
            axs[1].plot(time_steps, velocities_x[i], label=f'UAV {i}')
        else:
            axs[1].plot(time_steps, velocities_x[i], label=f'Target')
    axs[1].set_title('Velocity X Component vs Time')
    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel('Velocity X Component')
    axs[1].legend()

    for i in range(len(velocities_y)):
        if i != 3:
            axs[2].plot(time_steps, velocities_y[i], label=f'UAV {i}')
        else:
            axs[2].plot(time_steps, velocities_y[i], label=f'Target')
    axs[2].set_title('Velocity Y Component vs Time')
    axs[2].set_xlabel('Time Step')
    axs[2].set_ylabel('Velocity Y Component')
    axs[2].legend()

    for i in range(len(velocities_z)):
        if i != 3:
            axs[3].plot(time_steps, velocities_z[i], label=f'UAV {i}')
        else:
            axs[3].plot(time_steps, velocities_z[i], label=f'Target')
    axs[3].set_title('Velocity Z Component vs Time')
    axs[3].set_xlabel('Time Step')
    axs[3].set_ylabel('Velocity Z Component')
    axs[3].legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    env = UAVEnv()
    n_agents = env.num_agents
    n_actions = 3  # 3D actions (ax, ay, az)
    actor_dims = []
    velocities_magnitude = [[] for _ in range(env.num_agents)]  # record magnitude of vel
    velocities_x = [[] for _ in range(env.num_agents)]  # record vel_x
    velocities_y = [[] for _ in range(env.num_agents)]  # record vel_y
    velocities_z = [[] for _ in range(env.num_agents)]  # record vel_z

    for agent_id in env.observation_space.keys():
        actor_dims.append(env.observation_space[agent_id].shape[0])
    critic_dims = sum(actor_dims)
    # 3D observation dims: hunter=47, target=41 (derived from env.observation_space)
    obs_agt = actor_dims[0]
    obs_tar = actor_dims[-1]
    maddpg_agents = MADDPGWithAttention(actor_dims, critic_dims, n_agents, n_actions,
                           obs_agt=obs_agt, obs_tar=obs_tar,
                           alpha=0.0001, beta=0.003, scenario='UAV_Round_up',
                           chkpt_dir='tmp/maddpgwithatt/')

    try:
        maddpg_agents.load_checkpoint()
    except Exception as e:
        print(f'[warn] load_checkpoint skipped (no/incompatible checkpoint): {e}')
    print('---- Evaluating ----')

    obs = env.reset()

    def update(frame):
        global obs,velocities_magnitude,velocities_x,velocities_y,velocities_z

        for i in range(env.num_agents):
            vel = env.multi_current_vel[i]
            v_x, v_y, v_z = vel[:3]
            speed = np.linalg.norm(vel)

            velocities_magnitude[i].append(speed)
            velocities_x[i].append(v_x)
            velocities_y[i].append(v_y)
            velocities_z[i].append(v_z)

        actions = maddpg_agents.choose_action(obs, total_steps, evaluate=True)
        obs_, _, dones = env.step(actions)
        env.render_anime(frame)
        obs = obs_
        if any(dones) or frame > 1000:
            ani.event_source.stop()
            print("Round-up finished in",frame,"steps.")
            # smoothed_velocities_magnitude = [[] for _ in range(env.num_agents)]
            # smoothed_velocities_x = [[] for _ in range(env.num_agents)]
            # smoothed_velocities_y = [[] for _ in range(env.num_agents)]
            # smoothed_velocities_z = [[] for _ in range(env.num_agents)]
            # for i in range(env.num_agents):
            #     _velocity_magnitude = moving_average(velocities_magnitude[i],window_size=5)
            #     _velocity_x = moving_average(velocities_x[i],window_size=5)
            #     _velocity_y = moving_average(velocities_y[i],window_size=5)
            #     _velocity_z = moving_average(velocities_z[i],window_size=5)
            #     smoothed_velocities_magnitude[i]=_velocity_magnitude
            #     smoothed_velocities_x[i]=_velocity_x
            #     smoothed_velocities_y[i]=_velocity_y
            #     smoothed_velocities_z[i]=_velocity_z
            # smoothed_velocities_z = [moving_average(v, window_size=5) for v in velocities_z]
            # # plot_velocities(smoothed_velocities_magnitude,smoothed_velocities_x,smoothed_velocities_y,smoothed_velocities_z)
            # time_steps = range(len(smoothed_velocities_magnitude[0]))
            # plot_velocity_magnitude(time_steps,smoothed_velocities_magnitude)
            # plot_velocity_x(time_steps, smoothed_velocities_x)
            # plot_velocity_y(time_steps, smoothed_velocities_y)
        return []

    total_steps = 0

    fig = plt.figure()
    ani = animation.FuncAnimation(fig, update, frames=10000, interval=20)
    plt.show()