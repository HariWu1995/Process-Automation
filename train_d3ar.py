#!/usr/bin/env python
# coding: utf-8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

dir_path, file_path = os.path.realpath(__file__).rsplit("\\", 1)
print(dir_path, '-->', file_path)

import sys
import subprocess

import csv
import json

import time
import datetime

from tqdm import tqdm
from glob import glob
from copy import deepcopy

import cv2 
import math
import numpy as np

import matplotlib.pyplot as plt

from Agents.agent_d3ar import *
from Environments.environment_d3ar import *
from wealth_of_experience.experience_buffer import *


def clear_cmd():
    flag = os.system('clear' if os.name =='posix' else 'cls')


def onTraining(trainMore, **kwargs):
    trainMore = bool(trainMore)
    if set(['timeOut', 'timer']).issubset(set(kwargs.keys())):
        trainMore = trainMore and bool(kwargs['timer']<kwargs['timeOut'])
    if set(['maxEpisode', 'currentEpisode']).issubset(set(kwargs.keys())):
        trainMore = trainMore and bool(kwargs['currentEpisode']<kwargs['maxEpisode'])
    return trainMore


clear_cmd()

# Read the configuration file 
with open('RL_config_UTL_TM.json', 'r') as reader:
    Config = json.load(reader)

# Assign training hyper-parameters with configuration
PROCESS = Config['Process']
TIME_OUT = Config['TimeOut']
MAX_MOVES = Config['MaxMovesPerEpisode']
BATCH_SIZE = Config['BatchSizeForPeriodicTrain']
MAX_EPISODE = Config['MaxEpisode']
TRACE_LENGTH = Config['TraceLengthForPeriodicTrain']

TRAIN_ALL_PATHS = Config['TrainAllPaths'] # SHOULD be False in case of infinity-loop
MAX_PATHS_INIT = int(Config['MaxPathsInit'])
DUPLICATE_PATH = float(Config['DuplicatePaths']) if Config['DuplicatePaths'] else 1.0

# New params comparing to the last file
MemoryWindow = Config['MemoryWindow']
AugmentRate = Config['AugmentedPositionRate']
RESOLUTION = (Config['SCREEN_HEIGHT'], Config['SCREEN_WIDTH'])

useText = Config['UseText']
useIcon = Config['UseIcon']
useTemplate = Config['UseTemplate']
arrivalNode = Config['ArrivalNode']
destinationNode = Config['DestinationNode']
nodeTypesArchieved = ['actor', 'closer']
if Config['WalkOnNodesIncluded']:
    nodeTypesArchieved += ['walk-on']

optimizerQN = Config['OptimizerQN']
continueTrain = Config['ContinueTraining']
maxModelsStored = Config['MaxStoredModels']
winningSaturation = int(Config['WinningSaturation']) 

gama = Config['FutureRewardRate']
epsilon = Config['Epsilon']
alpha_1 = Config['LearningRateForRightPath']
alpha_2 = Config['LearningRateForWrongPath']
lossLimit = Config['LossUpperLimit']
lossInitial = Config['LossInitialForLearningRate']
guideFactor = Config['FollowGuidesBaseNumber']

save_freq = Config['PeriodForSaveModel'] # How often to save weights
store_freq = Config['PeriodForStoreExperience'] # How often to store EXP.
train_freq = Config['PeriodForTrainBatch'] # How often to practice a training
update_freq_1 = Config['PeriodForUpdateTargetBefore1rstWin'] # How often to update Target-network
update_freq_2 = Config['PeriodForUpdateTargetAfter1rstWin'] 


""" Load ENVIRONMENT and AGENT and EXPERIENCE_WEALTH """
env = Environment(process=PROCESS, 
                  use_text=useText, 
                  use_icon=useIcon,
                  use_template=useTemplate,
                  MAX_MOVES=MAX_MOVES,
                  MemoryWindow=MemoryWindow,
                  augmentRate=AugmentRate)
env.load_data()
env.set_arrival_and_destination(arrival_id=int(arrivalNode), 
                            destination_id=int(destinationNode))
TOTAL_PATHS = env.get_all_paths()
# max_moves = len(max(TOTAL_PATHS, key=len))
TOTAL_NODES = env.get_all_nodes(nodeTypesArchieved)
TOTAL_NODES = set(TOTAL_NODES)


agent = Agent(process=PROCESS, 
              use_text=useText, 
              use_icon=useIcon,
              use_template=useTemplate,
              augmentRate=AugmentRate,
              resolution=RESOLUTION,
              contexts=env.contexts, 
              actions_list=env.actions_list,
              optimizer_option=optimizerQN,
              n_models=int(maxModelsStored),
              reTrain=continueTrain,
              gama=gama,
              mode="train")

env.PosEnc = deepcopy(agent.PosEnc)
env.class_mapper = deepcopy(agent.class_mapper)

experience_wealth = ExperienceWealth(process=PROCESS, 
                                     exp_shape=9, 
                                     medium_traces=len(env.get_all_nodes()))

# Check hyper-params before training
print('\n\n\n'+'-'*37)
print('\n'.join('{} - {}'.format(k,v) for k,v in Config.items()))
print('\n'+'-'*37)
# print('Max moves:', max_moves)
print('Total nodes:', TOTAL_NODES)
print('Number of paths:', len(TOTAL_PATHS))
print('-'*37+'\n\n\n')
_ = input("Please check all the above training parameters!\n"+\
          "In case of any unsatisfaction, press Ctrl+C.\n"+\
          "Otherwise, press any key to start training ")

""" Training the network """
try:
    print("\n\n\nTraining ...\n\n\n")

    # Training-parameters
    nodes_archieved = set()
    winning_streak = 0
    paths_found = []
    update_freq = update_freq_1
    last_moves = 0
    N_paths = 0
    episode = 0
    loss = lossInitial

    start = time.time()
    WIN_FLAG = False
    MAX_PATHS = MAX_PATHS_INIT
    N_FINISHES = 0
    SELF_LEARNING = True
    while onTraining(SELF_LEARNING, currentEpisode=episode, maxEpisode=MAX_EPISODE):
        episode += 1

        # Initialize logs
        train_logs = dict()
        N_RIGHTs = 0
        N_WRONGs = 0

        if WIN_FLAG:
            update_freq = update_freq_2

        temp_buffer = []

        # Reset environment and agent
        env.reset()
        agent.be_ready()

        # Get 1st observation
        current_tokens, current_positions = env.observe()

        """ Experience with the Environment """
        current_moves = 0
        FOLLOW_INSTRUCTIONS = False
        while current_moves<MAX_MOVES and not (env.game_over or env.end_game): 
            current_moves += 1

            # Query action-proposal from Q-network
            action_batch, previous_contexts = agent.query(current_tokens, current_positions)

            # Check whether proposal is right or wrong
            if action_batch[0] == env.instruction_id:
                N_RIGHTs += 1
            else:
                N_WRONGs += 1

            # Decide to follow proposal above or guide from ENVIRONMENT
            N_FINISHES = max(0, N_FINISHES)
            followGuide = random.random() < guideFactor**(last_moves+N_FINISHES+N_paths*2)
            if followGuide:
                action_id = env.instruction_id
                FOLLOW_INSTRUCTIONS = True
                IMITIATE_BOOL = True
            else:
                action_id = action_batch[0]
                IMITIATE_BOOL = False
        
            action = agent.actions_list[action_id]

            """ AGENT practices an ACTION to the ENVIRONMENT and get REWARD """
            previous_node = env.node_id

            reward = env.next_state(by_action=action)
            current_node = env.node_id

            # Assign the old observation & Get the new one
            previous_tokens, previous_positions = current_tokens, current_positions
            current_tokens, current_positions = env.observe()

            # Update current experience
            temp_buffer.append(np.asarray([previous_tokens,
                                           previous_positions,
                                           previous_contexts[0],
                                           current_tokens,
                                           current_positions,
                                           agent.contexts_stack[0],
                                           env.end_game, 
                                           action_id, 
                                           reward]))
        last_moves = current_moves
        """ End of Experience """

        # Update flexible training params
        if env.end_game:
            WIN_FLAG = True
            N_FINISHES += 1
            winning_streak += 1

            # Update the paths have been found
            if not FOLLOW_INSTRUCTIONS \
            and env.archieved_nodes not in paths_found:
                paths_found += [env.archieved_nodes]
                nodes_archieved = set(
                    list(nodes_archieved)+env.archieved_nodes
                )

        # Record paths and check finish
        if (env.game_over and N_paths>=MAX_PATHS) or \
        (env.end_game and winning_streak/MAX_PATHS>winningSaturation):

            MAX_PATHS = N_paths

            # Record which nodes have been archieved
            ckpt_path = 'WIN-{}-{}-{:0>7d}.ckpt'.format(N_paths, winning_streak, episode)
            agent.save_session(ckpt_path)

            paths_recorder = open(
                os.path.join(agent.saver_dir, "paths.txt"), 'a')
            paths_recorder.write("Model = {}\n".format(ckpt_path))
            paths_recorder.write("\tTime = {}\n".format(time.time()-start))
            paths_recorder.writelines('\n'.join(str(path) for path in paths_found))
            paths_recorder.write('\n\t--> '+str(nodes_archieved))
            paths_recorder.write('\n'+'-'*37+'\n')
            paths_recorder.close()

            # Check whether to finish training
            if nodes_archieved == TOTAL_NODES:
                print("\n\n\nVictory!!!\n")
                print("[Nodes Archieved]", nodes_archieved, '\n\n\n')
                SELF_LEARNING = False

                if TRAIN_ALL_PATHS and N_paths<len(TOTAL_PATHS)*DUPLICATE_PATH:
                    print("False Alarm - Not Enough PATHs Yet")
                    SELF_LEARNING = True

            elif len(list(nodes_archieved)) < len(list(TOTAL_NODES))//2:
                WIN_FLAG = False

            path_siblings = glob(agent.saver_dir+'\\WIN-{}-*.ckpt.meta'.format(MAX_PATHS))
            if len(path_siblings) > winningSaturation//2:
                print("Learning is saturated at {} paths after {} continuous episodes".format(MAX_PATHS, winningSaturation))
                SELF_LEARNING = False
                
        N_paths = len(paths_found)
        N_nodes = len(list(nodes_archieved))

        # Update logs
        train_logs['N_actions_right'] = N_RIGHTs
        train_logs['N_actions_wrong'] = N_WRONGs
        train_logs['N_actions'] = current_moves
        train_logs['N_paths'] = N_paths
        train_logs['N_nodes'] = N_nodes
        train_logs['N_wins'] = winning_streak
        train_logs['loss'] = loss
        agent.update_logger(episode, train_logs)

        # Add the current experience to the wealth         
        episode_exp = list(zip(temp_buffer))
        experience_wealth.gather_batch(episode_exp)

        # Store experience into wealth
        if episode%store_freq==0 and len(experience_wealth.wealth)>16:
            experience_wealth.store(suffix=episode)

        # Update the Target-network
        if episode%update_freq==update_freq//2:
            agent.update_Target()

        # Periodically save the model
        if episode%save_freq==0 and MAX_PATHS>1:
            ckpt_path = 'model-{:0>7d}.ckpt'.format(episode)
            agent.save_session(ckpt_path)
            print("\nSaved Model @", ckpt_path)
                
        # Update flexible training params
        if env.game_over:
            N_FINISHES -= 0.3
            paths_found = []
            winning_streak = 0
            nodes_archieved = set()

        # Do NOT train if model still wins
        if (env.end_game and not FOLLOW_INSTRUCTIONS) or not SELF_LEARNING:
            continue

        """ TRAINING: Start """
        # Prepare very-presently experience for train
        batch_size = 1
        trace_length = len(episode_exp)
        trainData = np.asarray(episode_exp)
        trainBatch = trainData.reshape([trace_length, 
                                        experience_wealth.exp_shape])

        # Overwrite experience if it's turn to train batch
        # if episode%train_freq==0 and len(experience_wealth.wealth)>128:
        #     batch_size = BATCH_SIZE
        #     trace_length = min(TRACE_LENGTH, max(1, len(episode_exp)-1))
        #     trainBatch = experience_wealth.sample(batch_size, trace_length)

        # Clip-off the loss and transform the learning rate more adaptively
        sample_loss = lossLimit if np.isnan(loss) else min(abs(loss), lossLimit)
        alpha = alpha_1 if WIN_FLAG else alpha_2
        learning_rate = alpha*math.exp(sample_loss) / (1+math.exp(sample_loss))

        # Train Primary Q-Network
        loss = agent.train_Primary(trainBatch, learning_rate)
        
        """ TRAINING: Stop """ 

# except (KeyboardInterrupt, IndexError) as e:
except KeyboardInterrupt as e:
    print(e)


cmd = input("\n[ESCAPE] Press ANY key to continue ...")
if cmd.lower() not in ["q", "c", "quit", "cancel"]:
    experience_wealth.store(suffix=episode)

    ckpt_path = 'model-{:0>7d}.ckpt'.format(episode)
    agent.save_session(ckpt_path)
    print("\nSaved Model @", ckpt_path)






