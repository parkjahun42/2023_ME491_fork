from ruamel.yaml import YAML, dump, RoundTripDumper
from ME491_2023_project.env.RaisimGymVecEnvTrain import RaisimGymVecEnvTrain as VecEnv
from ME491_2023_project.helper.raisim_gym_helper import ConfigurationSaver, load_param, load_opponent_param, tensorboard_launcher
# from ME491_2023_project.env.bin.rsg_anymal import NormalSampler
from ME491_2023_project.env.RewardAnalyzer import RewardAnalyzer
from ME491_2023_project.env.bin.SI20233319vsSI99999999 import RaisimGymEnvTrain, NormalSampler
import os
import math
import time
import ME491_2023_project.algo.ppo.module as ppo_module
import ME491_2023_project.algo.ppo.ppo as PPO
from importlib import import_module
import torch.nn as nn
import numpy as np
import torch
import datetime
import argparse
import re

# task specification
task_name = "ME491_2023_project"
task_path = os.path.dirname(os.path.realpath(__file__))
# import module from the built environment library
# files = os.listdir(task_path)
# pattern = re.compile(r'AnymalControllerTrain_(\d+).hpp')
# for file in files:
#     match = pattern.match(file)
#     if match:
#         student_id = match.group(1)
#         print(student_id)
# module_name = f"ME491_2023_project.env.bin.SI20233319vsSI99999999"
# module = import_module(module_name)


# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', help='set mode either train or test', type=str, default='train')
parser.add_argument('-w', '--weight', help='pre-trained weight path', type=str, default='')
parser.add_argument('-op', '--opponent', help='pre-trained opponent weight path', type=str, default='')
parser.add_argument('-op2', '--opponent2', help='pre-trained opponent weight path', type=str, default='')
parser.add_argument('-op3', '--opponent3', help='pre-trained opponent weight path', type=str, default='')
args = parser.parse_args()
mode = args.mode
weight_path = args.weight
opponent_weight_path = args.opponent
opponent_weight_path2 = args.opponent2
opponent_weight_path3 = args.opponent3

# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../.."

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

is_pretrain = cfg['training']['is_pretrain']

# create environment from the configuration file
env = VecEnv(RaisimGymEnvTrain(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)))
env.seed(cfg['seed'])

# shortcuts
ob_dim = env.num_obs
opponent_ob_dim = env.opponent_num_obs
act_dim = env.num_acts
num_threads = cfg['environment']['num_threads']

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
total_steps = n_steps * env.num_envs



avg_rewards = []

actor = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim, act_dim),
                         ppo_module.MultivariateGaussianDiagonalCovariance(act_dim,
                                                                           env.num_envs,
                                                                           5.0,
                                                                           NormalSampler(act_dim),
                                                                           cfg['seed']),
                         device)
critic = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], nn.LeakyReLU, ob_dim, 1),
                           device)
opponent_actor = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, opponent_ob_dim, act_dim),
                                  ppo_module.MultivariateGaussianDiagonalCovariance(act_dim,
                                                                                    env.num_envs,
                                                                                    5.0,
                                                                                    NormalSampler(act_dim),
                                                                                    cfg['seed']),
                                  device)

opponent_actor2 = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, opponent_ob_dim, act_dim),
                                  ppo_module.MultivariateGaussianDiagonalCovariance(act_dim,
                                                                                    env.num_envs,
                                                                                    5.0,
                                                                                    NormalSampler(act_dim),
                                                                                    cfg['seed']),
                                  device)

opponent_actor3 = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, opponent_ob_dim, act_dim),
                                  ppo_module.MultivariateGaussianDiagonalCovariance(act_dim,
                                                                                    env.num_envs,
                                                                                    5.0,
                                                                                    NormalSampler(act_dim),
                                                                                    cfg['seed']),
                                  device)

if is_pretrain:
    saver = ConfigurationSaver(log_dir=home_path + "/ME491_2023_project/data/" + task_name, save_items=[task_path + "/cfg.yaml", task_path + "/runner.py", task_path + "/EnvironmentPre.hpp", task_path + "/PretrainingAnymalController_20233319.hpp"])
else:
    saver = ConfigurationSaver(log_dir=home_path + "/ME491_2023_project/data/" + task_name, save_items=[task_path + "/cfg.yaml", task_path + "/runnerTrain.py", task_path + "/EnvironmentTrain.hpp", task_path + "/AnymalControllerTrain_20233319.hpp", task_path + "/AnymalControllerTrain_99999999.hpp"])
tensorboard_launcher(saver.data_dir+"/..")  # press refresh (F5) after the first ppo update

ppo = PPO.PPO(actor=actor,
              critic=critic,
              num_envs=cfg['environment']['num_envs'],
              num_transitions_per_env=n_steps,
              num_learning_epochs=4,
              gamma=0.95,
              lam=0.95,
              num_mini_batches=4,
              device=device,
              log_dir=saver.data_dir,
              shuffle_batch=False,
              )

reward_analyzer = RewardAnalyzer(env, ppo.writer)

if mode == 'retrain':
    load_param(weight_path, env, actor, critic, ppo.optimizer, saver.data_dir)
if not is_pretrain and opponent_weight_path != '':
    load_opponent_param(opponent_weight_path, env, opponent_actor, saver.data_dir, 1)
    load_opponent_param(opponent_weight_path2, env, opponent_actor2, saver.data_dir, 2)
    load_opponent_param(opponent_weight_path3, env, opponent_actor3, saver.data_dir, 3)

num_envs = cfg['environment']['num_envs']
check_me_first = False
for update in range(1000000):
    start = time.time()
    reward_sum = 0
    done_sum = 0
    average_dones = 0.

    env.check_curriculum()
    mode = env.mode_callback()
    modeLevel = env.mode_level_callback()
    currLevel = env.get_curriculum_level()
    modeCurr0 = currLevel[:(int)(num_envs * (mode[0]))].mean()
    modeCurr1 = currLevel[(int)(num_envs * mode[0]):(int)(num_envs * (mode[0] + mode[1]))].mean()
    modeCurr2 = currLevel[(int)(num_envs * (mode[0] + mode[1])):(int)(num_envs * (mode[0] + mode[1] + mode[2]))].mean()
    modeCurr3 = currLevel[(int)(num_envs * (mode[0] + mode[1] + mode[2])):(int)(num_envs * (mode[0] + mode[1] + mode[2] + mode[3]))].mean()
    modeCurr4 = currLevel[(int)(num_envs * (mode[0] + mode[1] + mode[2] + mode[3])):(int)(num_envs * (mode[0] + mode[1] + mode[2] + mode[3] + mode[4]))].mean()
    if modeLevel == 1 and modeCurr2 > 100:
        load_opponent_param(opponent_weight_path2, env, opponent_actor2, saver.data_dir, 2, 4000)
    if modeLevel > 1:
        if check_me_first == False:
            load_opponent_param(saver.data_dir+"/full_"+str(update)+'.pt', env, opponent_actor, saver.data_dir, 1, update)
            check_me_first = True
        else:
            if(update % 100 == 0 and update > 1000):
                load_opponent_param(saver.data_dir+"/full_"+str(update)+'.pt', env, opponent_actor, saver.data_dir, 1, update-1000)

    if update % cfg['environment']['eval_every_n'] == 0:
        env.reset()
        print("Visualizing and evaluating the current policy")
        torch.save({
            'actor_architecture_state_dict': actor.architecture.state_dict(),
            'actor_distribution_state_dict': actor.distribution.state_dict(),
            'critic_architecture_state_dict': critic.architecture.state_dict(),
            'optimizer_state_dict': ppo.optimizer.state_dict(),
        }, saver.data_dir+"/full_"+str(update)+'.pt')
        # we create another graph just to demonstrate the save/load method
        loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim, act_dim)
        loaded_graph.load_state_dict(torch.load(saver.data_dir+"/full_"+str(update)+'.pt')['actor_architecture_state_dict'])

        env.turn_on_visualization()
        env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_"+str(update)+'.mp4')

        for step in range(n_steps*2):
            with torch.no_grad():
                frame_start = time.time()
                if is_pretrain:
                    obs = env.observe(False)
                    action = loaded_graph.architecture(torch.from_numpy(obs).cpu()).detach()
                    reward, dones = env.step(action.cpu().detach().numpy())
                else:
                    obs, opponent_obs = env.observe(False)
                    action = loaded_graph.architecture(torch.from_numpy(obs).cpu())
                    opponent_action = torch.zeros_like(action)
                    opponent_action[(int)(num_envs * mode[0]):(int)(num_envs * (mode[0] + mode[1]))] = opponent_actor.noiseless_action(opponent_obs[(int)(num_envs * mode[0]):(int)(num_envs * (mode[0] + mode[1]))]).detach()
                    opponent_action[(int)(num_envs * (mode[0] + mode[1])):(int)(num_envs * (mode[0] + mode[1] + mode[2]))] = opponent_actor2.noiseless_action(opponent_obs[(int)(num_envs * (mode[0] + mode[1])):(int)(num_envs * (mode[0] + mode[1] + mode[2]))]).detach()
                    opponent_action[(int)(num_envs * (mode[0] + mode[1] + mode[2])):(int)(num_envs * (mode[0] + mode[1] + mode[2] + mode[3]))] = opponent_actor3.noiseless_action(opponent_obs[(int)(num_envs * (mode[0] + mode[1] + mode[2])):(int)(num_envs * (mode[0] + mode[1] + mode[2] + mode[3]))]).detach()

                    # opponent_action = opponent_actor.noiseless_action(opponent_obs)
                    reward, dones = env.step(action.cpu().detach().numpy(), opponent_action.cpu().detach().numpy())

                # reward_analyzer.add_reward_info(env.get_reward_info())
                frame_end = time.time()
                wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
                if wait_time > 0.:
                    time.sleep(wait_time)

        env.stop_video_recording()
        env.turn_off_visualization()

        # reward_analyzer.analyze_and_plot(update)
        env.reset()
        env.save_scaling(saver.data_dir, str(update))

    # actual training
    if update % 20 < 2:
        env.turn_on_visualization()
    else:
        env.turn_off_visualization()
    for step in range(n_steps):
        if is_pretrain:
            obs = env.observe()
            action = ppo.act(obs)
            # opponent_action = ppo.act(opponent_obs)
            reward, dones = env.step(action)
        else:
            obs, opponent_obs = env.observe()
            action = ppo.act(obs)
            opponent_action = torch.zeros_like(torch.from_numpy(action))
            opponent_action[(int)(num_envs * mode[0]):(int)(num_envs * (mode[0] + mode[1]))] = opponent_actor.noiseless_action(opponent_obs[(int)(num_envs * mode[0]):(int)(num_envs * (mode[0] + mode[1]))]).detach()
            opponent_action[(int)(num_envs * (mode[0] + mode[1])):(int)(num_envs * (mode[0] + mode[1] + mode[2]))] = opponent_actor2.noiseless_action(opponent_obs[(int)(num_envs * (mode[0] + mode[1])):(int)(num_envs * (mode[0] + mode[1] + mode[2]))]).detach()
            opponent_action[(int)(num_envs * (mode[0] + mode[1] + mode[2])):(int)(num_envs * (mode[0] + mode[1] + mode[2] + mode[3]))] = opponent_actor3.noiseless_action(opponent_obs[(int)(num_envs * (mode[0] + mode[1] + mode[2])):(int)(num_envs * (mode[0] + mode[1] + mode[2] + mode[3]))]).detach()

            opponent_action = opponent_action.detach().numpy()
            reward, dones = env.step(action, opponent_action)

        if update % cfg['environment']['analyze_freq'] == 0:
            reward_analyzer.add_reward_info(env.get_reward_info())

        ppo.step(value_obs=obs, rews=reward, dones=dones)
        done_sum = done_sum + np.sum(dones)
        reward_sum = reward_sum + np.sum(reward)

    if update % cfg['environment']['analyze_freq'] == 0:
        reward_analyzer.analyze_and_plot(update)

    # take st step to get value obs
    if is_pretrain:
        obs = env.observe()
    else:
        obs, opponent_obs = env.observe()
    ppo.update(actor_obs=obs, value_obs=obs, log_this_iteration=update % 10 == 0, update=update)
    average_ll_performance = reward_sum / total_steps
    average_dones = done_sum / total_steps
    avg_rewards.append(average_ll_performance)

    actor.update()
    actor.distribution.enforce_minimum_std((torch.ones(12)*0.2).to(device))

    # curriculum update. Implement it in Environment.hpp
    env.curriculum_callback(update)

    end = time.time()


    print('----------------------------------------------------')
    print('{:>6}th iteration'.format(update))
    print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(average_ll_performance)))
    print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
    print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
    print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps / (end - start))))
    print('{:<40} {:>6}'.format("real time factor: ", '{:6.0f}'.format(total_steps / (end - start)
                                                                       * cfg['environment']['control_dt'])))
    print('{:<40} {:>6}'.format("mode level: ", '{:6.0f}'.format(modeLevel)))
    print('{:<1} {:>0} {:<1} {:>0} {:<1} {:>0} {:<1} {:>0} {:<1} {:>0}'.format("mode prob 0: ", '{:3.2f}'.format(mode[0]), "1: ", '{:3.2f}'.format(mode[1]), "2: ", '{:3.2f}'.format(mode[2]), "3: ", '{:3.2f}'.format(mode[3]), "4: ", '{:3.2f}'.format(mode[4])))
    print('{:<1} {:>0} {:<1} {:>0} {:<1} {:>0} {:<1} {:>0} {:<1} {:>0}'.format("Curr level 0: ", '{:3.2f}'.format(modeCurr0), "1: ", '{:3.2f}'.format(modeCurr1), "2: ", '{:3.2f}'.format(modeCurr2), "3: ", '{:3.2f}'.format(modeCurr3), "4: ", '{:3.2f}'.format(modeCurr4)))
    print('----------------------------------------------------\n')
