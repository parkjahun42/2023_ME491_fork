# //----------------------------//
# // This file is part of RaiSim//
# // Copyright 2020, RaiSim Tech//
# //----------------------------//

import numpy as np
import platform
import os


class RaisimGymVecEnvTrain:

    def __init__(self, impl, normalize_ob=True, seed=0, clip_obs=10.):
        if platform.system() == "Darwin":
            os.environ['KMP_DUPLICATE_LIB_OK']='True'
        self.normalize_ob = normalize_ob
        self.clip_obs = clip_obs
        self.wrapper = impl
        self.num_obs = self.wrapper.getObDim()
        self.opponent_num_obs = self.wrapper.getOpponentObDim()
        self.num_acts = self.wrapper.getActionDim()
        self.num_mode = self.wrapper.getModeNum()
        self.curriculum_level = np.zeros(self.num_envs, dtype=np.int32)
        self._mode = np.zeros(self.num_mode, dtype=np.float32)
        self._observation = np.zeros([self.num_envs, self.num_obs], dtype=np.float32)
        self._opponent_observation = np.zeros([self.num_envs, self.opponent_num_obs], dtype=np.float32)
        self.actions = np.zeros([self.num_envs, self.num_acts], dtype=np.float32)
        self.log_prob = np.zeros(self.num_envs, dtype=np.float32)
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self._done = np.zeros(self.num_envs, dtype=bool)
        self.rewards = [[] for _ in range(self.num_envs)]
        self.wrapper.setSeed(seed)
        self.modeLevel = self.wrapper.getModeLevel()
        self.count = 0.0
        self.mean = np.zeros(self.num_obs, dtype=np.float32)
        self.var = np.zeros(self.num_obs, dtype=np.float32)

    def seed(self, seed=None):
        self.wrapper.setSeed(seed)

    def turn_on_visualization(self):
        self.wrapper.turnOnVisualization()

    def turn_off_visualization(self):
        self.wrapper.turnOffVisualization()

    def start_video_recording(self, file_name):
        self.wrapper.startRecordingVideo(file_name)

    def stop_video_recording(self):
        self.wrapper.stopRecordingVideo()

    def step(self, action, opponent_action):
        self.wrapper.step(action, opponent_action, self._reward, self._done)
        return self._reward.copy(), self._done.copy()

    def load_scaling(self, dir_name, iteration, count=1e5):
        mean_file_name = dir_name + "/mean" + str(iteration) + ".csv"
        var_file_name = dir_name + "/var" + str(iteration) + ".csv"
        self.count = count
        self.mean = np.loadtxt(mean_file_name, dtype=np.float32)
        self.var = np.loadtxt(var_file_name, dtype=np.float32)
        self.wrapper.setObStatistics(self.mean, self.var, self.count)

    def load_opponent_scaling(self, dir_name, iteration, count=1e5):
        mean_file_name = dir_name + "/mean" + str(iteration) + ".csv"
        var_file_name = dir_name + "/var" + str(iteration) + ".csv"
        self.opponent_count = count
        self.opponent_mean = np.loadtxt(mean_file_name, dtype=np.float32)
        self.opponent_var = np.loadtxt(var_file_name, dtype=np.float32)
        self.wrapper.setOpponentObStatistics(self.opponent_mean, self.opponent_var, self.opponent_count)

    def load_opponent_scaling2(self, dir_name, iteration, count=1e5):
        mean_file_name = dir_name + "/mean" + str(iteration) + ".csv"
        var_file_name = dir_name + "/var" + str(iteration) + ".csv"
        self.opponent_count2 = count
        self.opponent_mean2 = np.loadtxt(mean_file_name, dtype=np.float32)
        self.opponent_var2 = np.loadtxt(var_file_name, dtype=np.float32)
        self.wrapper.setOpponentObStatistics2(self.opponent_mean2, self.opponent_var2, self.opponent_count2)

    def load_opponent_scaling3(self, dir_name, iteration, count=1e5):
        mean_file_name = dir_name + "/mean" + str(iteration) + ".csv"
        var_file_name = dir_name + "/var" + str(iteration) + ".csv"
        self.opponent_count3 = count
        self.opponent_mean3 = np.loadtxt(mean_file_name, dtype=np.float32)
        self.opponent_var3 = np.loadtxt(var_file_name, dtype=np.float32)
        self.wrapper.setOpponentObStatistics3(self.opponent_mean3, self.opponent_var3, self.opponent_count3)

    def save_scaling(self, dir_name, iteration):
        mean_file_name = dir_name + "/mean" + iteration + ".csv"
        var_file_name = dir_name + "/var" + iteration + ".csv"
        self.wrapper.getObStatistics(self.mean, self.var, self.count)
        np.savetxt(mean_file_name, self.mean)
        np.savetxt(var_file_name, self.var)

    def observe(self, update_statistics=True):
        self.wrapper.observe(self._observation, self._opponent_observation, update_statistics)
        return self._observation, self._opponent_observation

    def get_reward_info(self):
        return self.wrapper.getRewardInfo()

    def reset(self):
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self.wrapper.reset()

    def close(self):
        self.wrapper.close()

    def curriculum_callback(self, iter):
        self.wrapper.curriculumUpdate(iter)

    def mode_callback(self):
        self.wrapper.modeUpdate(self._mode)
        return self._mode

    def mode_level_callback(self):

        return self.wrapper.getModeLevel()

    def check_curriculum(self):
        self.wrapper.checkCurriculum()

    def get_curriculum_level(self):
        self.wrapper.getCurrLevel(self.curriculum_level)
        return self.curriculum_level

    @property
    def num_envs(self):
        return self.wrapper.getNumOfEnvs()
