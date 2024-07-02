# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 16:15:27 2024

@author: aleks
"""
import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
from gym.spaces import Discrete


#just allow basic actions
class LimitedActionSpaceWrapper(gym.Wrapper):   #class used to limit the possible actions from 15 to 9, because the last 6 are key bind actions
    def __init__(self, env, allowed_actions):
        
        super(LimitedActionSpaceWrapper, self).__init__(env)
        self.allowed_actions = allowed_actions
        self.action_space = Discrete(len(allowed_actions))
        
    def step(self, action):
        mapped_action = self.allowed_actions[action]
        return self.env.step(mapped_action)
    
    def reset(self):
        return self.env.reset()
#define a replay buffer
class replayBuffer():
    def __init__(self, maxlen):  #input here are the length of the replay buffer,, stored actions
        self.buffer = deque([], maxlen = maxlen)
        
    def sample(self, sample_size): #takes inn the chosen sample size and returns a random sample of variables in that said size
        return random.sample(self.buffer, sample_size)
    
    def add(self, shift):  #takes in the statechange variables tuple and append them in the rightside of the buffer
        self.buffer.append(shift)
        
    def size(self): #return the length of the replay buffer
        return len(self.buffer)
    
def copy_weights(targetNN, policyNN):  #copy weights and biases from one NN to the other
    for target_param, source_param in zip(targetNN.parameters(), policyNN.parameters()):
        target_param.data.copy_(source_param.data)

#defining the DQN 
class DQNetwork(nn.Module):
    def __init__(self, in_shape, out_actions): #Inputs here are number of nodes for each of the layers
        super().__init__()
        
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels = in_shape, out_channels = 10, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 10, out_channels = 10, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            )   
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels= 10, out_channels =10, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 10, out_channels= 10, kernel_size= 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels= 10, out_channels =10, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 10, out_channels= 10, kernel_size= 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            )
        self.layer_out = nn.Sequential(
            nn.Flatten(),
            #Flatten the input from the conv layers and pass it to the action choosing output layer
            nn.Linear(in_features=10*8*8 , out_features= out_actions)
            )
        
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.layer_out(x)
        return x


class bigfishDQN():
    #hyperParameters
    learningRate = 0.0001  #alpha
    discountFactor = 0.9 #discount rate, gamma
    syncRate = 15 #steps before syncing target and policy nn
    bufferMaxlen = 200000 #max size of replay buffer
    sampledBatchSize = 64 #size of data sampled from the buffer    
    #Neural network
    lossFunction = nn.MSELoss() #MSE mean squared error
    optimizer =  None  #optimizer, choose Adam here
    
    allowed_actions = np.array([0,1,2,3,4,5,6,7,8])  #remove actions 9-14 since they also represent doing nothing 

    def train(self, episodes):
        
        seed = 1710
        epsilon = 1 #1 = 100% random actions
        epsilonList = [] #keep track over exploration
        epsilon_min = 0.05
        epsilon_decay = 4.75*10**(-5) #want to decay the amount of random actions over the first 20000 episodes
        steps = 0 #count to know when to sync
        trunc_steps = 0 #count to see if 
        max_steps = 25000 # after 25000 steps the truncated flag kick in, pretty high number but I think it is required
        rewardLog = np.zeros(episodes)
        render = True
        buffer = replayBuffer(self.bufferMaxlen)  #create the replay buffer with chosen maxlength
        
        env = gym.make("procgen:procgen-bigfish-v0", start_level=seed, num_levels=1, render_mode="rgb_array" if render else None, 
                        distribution_mode="easy", use_backgrounds=False )
        #get some information about the environment object 
        # print(f"the reward range for this environment is {env.reward_range}")
        # print(f"the spec for this environment is {env.spec}")
        # print(f"the metadata for this environment is {env.metadata}")
        wrap = LimitedActionSpaceWrapper(env, self.allowed_actions) #using the allowed actions
        
        #create NN for target and policy
        targetNN = DQNetwork(3, 9)
        policyNN = DQNetwork(3, 9)
        #policyNN.load_state_dict(torch.load("bigFish2306-test.pt"))  #load weights to continue training on already trained NN
        copy_weights(targetNN, policyNN)
        
        
        self.optimizer = torch.optim.Adam(policyNN.parameters(), lr=self.learningRate)

        for i in range(episodes): #iterate over all episodes
            state = env.reset()
            print(f"Now executing episode: {i}")
            #print(f"Initial state shape: {state.shape}")
            reward = 0
            trunc_steps = 0
            Terminated = False  
            Truncated = False   #more steps than max_steps will activate it
            #print(f"shape of observation space {env.observation_space}")
            while not Terminated and not Truncated:  #loop until a flag for each episode
                
                if epsilon >random.random():  #Check if we should explore
                    action = wrap.action_space.sample()  #choose a random action, (sample random action)
                    #action = random.random([0,1,2,3,4,5,6,7,8]) wrapper is a better way to do this, but this is essentially what happens
                else:
                    with torch.no_grad(): #save processing power by turning of the automatic calculated gradients
                        processedState = self.observation_to_input(state) #process the current state to input in NN
                        maxvalue = policyNN(processedState).argmax() #find the networks max value
                        #print(f"this is the max value and hence the action {maxvalue}")
                        action = maxvalue  #get the max value, this will be the action taken
                
                newState,reward_new,Terminated,_ = env.step(action)
                buffer.add((state, action, newState, Terminated, reward_new))
                #print(f"Old reward for episode is {reward}, new reward is {reward + reward_new +1}")
                reward += reward_new
                state = newState
                trunc_steps += 1
                steps += 1               
                if trunc_steps > max_steps:
                    print(f"Truncated, used more than {self.max_steps} steps")
                    Truncated = True
                    state = env.reset()
                if reward_new == 10:
                    print("episode completed sucsessfully")
                    state = env.reset()
            if Terminated or Truncated:
                #print(f"not good...")
                state = env.reset()            
            #print(f"this is the reward {reward} and this is the max reward {np.max(rewardLog)}")
            if reward > np.max(rewardLog):  #save other good policies to check training, might be problematic when the model solve the problem frequently?
                    torch.save(policyNN.state_dict(), f"bigFishMaxReward{reward}.pt")
            # Save the total rewards for the episode      
            if reward >= 1:
                rewardLog[i] = reward               
            # Check if enough data is in the replaybuffer
            if buffer.size() >= self.sampledBatchSize:                
                sampledBatch = buffer.sample(self.sampledBatchSize)

                current_q_list, target_q_list = self.learn(sampledBatch, policyNN, targetNN) #get the updated lists with q values from the policy network, and the calculated target q values
                loss = self.lossFunction(torch.stack(current_q_list), torch.stack(target_q_list)) #use the loss function and calculate the loss
                # Optimize the model
                self.optimizer.zero_grad()  #zero old gradients so that we only add the new ones
                loss.backward()  #calculate new gradients for loss change, 
                self.optimizer.step() #do a single step with the optimizer based on the calculated gradients, updates the parameters
               
            # Decay epsilon
            if epsilon >= epsilon_min:  #make sure there always are chosen some random actions
                epsilon = epsilon - epsilon_decay
                epsilonList.append(epsilon)

            # Copy values of policy network to target network after a certain number of steps
            if steps >= self.syncRate:
                copy_weights(targetNN, policyNN)
                steps = 0
        # Close environment
        env.close()
        # Save policy
        torch.save(policyNN.state_dict(), "bigFish-test123.pt")

    def qValueUpdate(self, step_reward, discount_factor, maxQ_nextState):
        q_new = (step_reward+discount_factor*maxQ_nextState)
            
        
        return q_new
        
        
        
    # make the agent learn by updating the q values
    def learn(self, Sampledbatch, policyNN, targetNN):
        current_q_list = []
        target_q_list = []
        
        for sample in Sampledbatch:
            state, action, newState, Terminated, reward_new = sample
            #print(f"action type is {}")
            current_q = policyNN(self.observation_to_input(state)) #get the current q values for the taken action, from policy network
            current_q_list.append(current_q) #append the q values from the policy network to the list
            maxQ_nextState = targetNN(self.observation_to_input(newState)).max()
            
            if Terminated:
                #if terminated the value should be either 0 or the total reward
                target = torch.FloatTensor([1.5* reward_new])  
            else:                
                # Calculate target q value with the dqn target formula
                targetvalue = self.qValueUpdate(reward_new, self.discountFactor, maxQ_nextState)
                with torch.no_grad():
                    target = torch.FloatTensor(targetvalue)
            if reward_new >= 1:
                target = target + reward_new  #increse the importance of eating fish by updating the q value further

            # Get the target set of Q values
            target_q = targetNN(self.observation_to_input(state)) # get the set of q values calculated by the target network for the action   
            # Adjust the specific action to the target that was just calculated. 
            
            #print(f"the target_q has the form {target_q}")
            target_q[0][action] = target
            target_q_list.append(target_q)
        return current_q_list, target_q_list
        
   
    def observation_to_input(self, state):
        #print(f"shape of state {state.shape}")  #Raw shape is (64,64,3)
        #Convert the state to a torch tensor and ensure the data type is float32
        input_tensor = torch.from_numpy(state).float()
        input_tensor = input_tensor.permute(2, 0, 1)
        # Add one more dimension representing batch, making the shape [1, 3, 64, 64]
        input_tensor = input_tensor.unsqueeze(0)
        return input_tensor

    # Run the environment with the learned policy
    def test(self, episodes):
        render = True
        env = gym.make("procgen:procgen-bigfish-v0", num_levels=0, render_mode="human" if render else None, 
                        distribution_mode="easy", use_backgrounds=False )
        # Load learned policy
        policyNN =  policyNN = DQNetwork(3, 9)
        policyNN.load_state_dict(torch.load("bigFishMaxReward42.pt"))
        policyNN.eval()    # switch model to evaluation mode
        rewardLog = np.zeros(episodes)
        for i in range(episodes):
            episodeReward = 0
            state = env.reset()  # Initialize to state 0
            terminated = False   # True when agent falls in hole or reached goal
            truncated = False    # True when agent takes more than 200 actions            

            # Agent navigates map until it falls into a hole (terminated), reaches goal (terminated), or has taken 200 actions (truncated).
            while(not terminated and not truncated):  
                # Select best action   
                with torch.no_grad():
                    processedState = self.observation_to_input(state) #process the current state
                    maxvalue = policyNN(processedState).argmax() #find the networks max value
                    action = maxvalue.item() #get the max as a python int

                # Execute action
                state,reward,terminated,_ = env.step(action)
                episodeReward += reward
            rewardLog[i] = episodeReward
        plt.plot(rewardLog, label = f"Reward over {episodes} episodes")    
        plt.show()
        avg = np.average(rewardLog)
        print(f"this is the average reward for the trained agent: {avg}")
        env.close()
        
def randomP(episodes):
    render = True
    env = gym.make("procgen:procgen-bigfish-v0", num_levels=0, render_mode="human" if render else None, 
                    distribution_mode="easy", use_backgrounds=False )
    rewardLog = np.zeros(episodes)
    for i in range(episodes):
        episodeReward = 0
        state = env.reset()  # Initialize to state 0
        terminated = False   # True when agent falls in hole or reached goal
        truncated = False    # True when agent takes more than 200 actions            

        # Agent navigates map until it falls into a hole (terminated), reaches goal (terminated), or has taken 200 actions (truncated).
        while(not terminated and not truncated):  
            action = env.action_space.sample()
            # Execute action
            state,reward,terminated,_ = env.step(action)
            episodeReward += reward
        rewardLog[i] = episodeReward
    plt.plot(rewardLog, label = f"Reward over {episodes} episodes")    
    plt.show()
    avg = np.average(rewardLog)
    print(f"this is the average reward for the random agent: {avg}")
    env.close()


bigFish = bigfishDQN()
bigFish.train(100)
#bigFish.test(50)
#randomP(50)


