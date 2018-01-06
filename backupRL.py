from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

import numpy as np
import random
import Queue
"""
Here are two values you can use to tune your Qnet
You may choose not to use them, but the training time
would be significantly longer.
Other than the inputs of each function, this is the only information
about the nature of the game itself that you can use.
"""
PIPEGAPSIZE  = 100
BIRDHEIGHT = 24
K = 400
k = 50
gamma = 0.9
record = []

class QNet(object):

    def __init__(self):
        """
        Initialize neural net here.
        You may change the values.

        Args:
            num_inputs: Number of nodes in input layer
            num_hidden1: Number of nodes in the first hidden layer
            num_hidden2: Number of nodes in the second hidden layer
            num_output: Number of nodes in the output layer
            lr: learning rate
        """
        self.num_inputs = 2
        self.num_hidden1 = 10
        self.num_hidden2 = 10
        self.num_output = 2
        self.lr = 0.001
        self.build()

    def build(self):
        """
        Builds the neural network using keras, and stores the model in self.model.
        Uses shape parameters from init and the learning rate self.lr.
        You may change this, though what is given should be a good start.
        """
        model = Sequential()
        model.add(Dense(self.num_hidden1, init='lecun_uniform', input_shape=(self.num_inputs,)))
        model.add(Activation('relu'))

        model.add(Dense(self.num_hidden2, init='lecun_uniform'))
        model.add(Activation('relu'))

        model.add(Dense(self.num_output, init='lecun_uniform'))
        model.add(Activation('linear'))

        rms = RMSprop(lr=self.lr)
        model.compile(loss='mse', optimizer=rms)
        self.model = model


    def flap(self, input_data):
        """
        Use the neural net as a Q function to act.
        Use self.model.predict to do the prediction.

        Args:
            input_data (Input object): contains information you may use about the 
            current state.

        Returns:
            (choice, prediction, debug_str): 
                choice (int) is 1 if bird flaps, 0 otherwise. Will be passed
                    into the update function below.
                prediction (array-like) is the raw output of your neural network,
                    returned by self.model.predict. Will be passed into the update function below.
                debug_str (str) will be printed on the bottom of the game
        """

        # state = your state in numpy array
        # prediction = self.model.predict(state.reshape(1, self.num_inputs), batch_size=1)[0]
        # choice = make choice based on prediction
        # debug_str = ""
        # return (choice, prediction, debug_str)

        state = np.array([input_data.distX,input_data.distY])
        prediction = self.model.predict(state.reshape(1,self.num_inputs), batch_size = 1)[0]
        print prediction
        choice = int(prediction[0] > prediction[1])
        # print choice

        return (choice,prediction,"")


    def update(self, last_input, last_choice, last_prediction, crash, scored, playerY, pipVelX):
        """
        Use Q-learning to update the neural net here
        Use self.model.fit to back propagate

        Args:
            last_input (Input object): contains information you may use about the
                input used by the most recent flap() 
            last_choice: the choice made by the most recent flap()
            last_prediction: the prediction made by the most recent flap()
            crash: boolean value whether the bird crashed
            scored: boolean value whether the bird scored
            playerY: y position of the bird, used for calculating new state
            pipVelX: velocity of pipe, used for calculating new state

        Returns:
            None
        """
        # This is how you calculate the new (x,y) distances
        # new_distX = last_input.distX + pipVelX
        # new_distY = last_input.pipeY - playerY

        # state = compute new state in numpy array
        # reward = compute your reward
        # prediction = self.model.predict(state.reshape(1, self.num_inputs), batch_size = 1)

        # update old prediction from flap() with reward + gamma * np.max(prediction)
        # record updated prediction and old state in your mini-batch
        # if batch size is large enough, back propagate
        # self.model.fit(old states, updated predictions, batch_size=size, epochs=1)
        def compute_reward(new_distX,new_distY,crash,scored,playerY):
            if scored:
                return 1000 #1000

            if new_distY < 0 or new_distY > PIPEGAPSIZE - BIRDHEIGHT - 0:
                return -100

            # if new_distX <= 50:
                # if new_distY <= 28 or new_distY >= 72:
                    # return -100
            
            anchor = playerY + new_distY - PIPEGAPSIZE / 2 + BIRDHEIGHT / 2


            reward = (PIPEGAPSIZE / 2 - abs(playerY - BIRDHEIGHT / 2 - anchor))
            # print new_distY, playerY, anchor, reward

            return reward

        global K,k,gamma,record
        new_distX = last_input.distX + pipVelX
        new_distY = last_input.pipeY - playerY
        
        state = np.array([new_distX,new_distY])

        prediction = self.model.predict(state.reshape(1, self.num_inputs), batch_size = 1)[0]

        # print prediction
        if crash:
            new = -10000
        else:
            new = compute_reward(new_distX,new_distY,crash,scored,playerY) + gamma * np.max(prediction)

        if last_choice == 1:
            new_prediction = np.array([new,last_prediction[1]])
        else:
            new_prediction = np.array([last_prediction[0],new])


        old_state = np.array([last_input.distX,last_input.distY])
        record.append([old_state,new_prediction])

        if len(record) == K:
            chosen = random.sample(record,k)
            # print chosen
            old_states = []
            updates = []
            # print chosen

            for data in chosen:
                old_states.append(data[0])
                updates.append(data[1])
            self.model.fit(np.array(old_states),np.array(updates),batch_size = k, epochs = 1,verbose = 0)

            del record[0]

            print "finished!"
        return 
        
class Input:
    def __init__(self, playerX, playerY, pipeX, pipeY,
                distX, distY):
        """
        playerX: x position of the bird
        playerY: y position of the bird
        pipeX: x position of the next pipe
        pipeY: y position of the next pipe
        distX: x distance between the bird and the next pipe
        distY: y distance between the bird and the next pipe
        """
        self.playerX = playerX
        self.playerY = playerY
        self.pipeX = pipeX
        self.pipeY = pipeY
        self.distX = distX
        self.distY = distY

