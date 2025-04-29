# IMPORTS
import sys
sys.path.append('/Applications/coppeliaSim.app/Contents/Resources/programming/zmqRemoteApi/clients/python/src') # CHANGE BEFORE RUNNING
import random
import numpy as np
import tensorflow as tf
import glob
import re
from collections import deque
from itertools import permutations
import cv2
import os
import time
import itertools
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

class DQNAgent:
    def __init__(self, state_size, action_size):
        # Initialize DQN parameters
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99 # Discount factor for future rewards
        self.epsilon = 0.5 # Starting exploration rate
        self.epsilon_min = 0.05 # Minimum exploration rate
        self.epsilon_decay = 0.95 # Decay rate for epsilon per episode
        self.learning_rate = 0.001 # Learning rate for the optimizer
        self.batch_size = 32 # Size of minibatch for training
        self.memory = deque(maxlen=50000) # Replay memory
        self.model = self._build_model() # Build the neural network model

    def _build_model(self):
        # Build a simple feedforward neural network
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_size,)),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear') # Output one Q-value per action
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        # Store experience in replay memory
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Choose an action: either random (exploration) or best (exploitation)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
        return np.argmax(q_values[0])

    def replay(self):
        # Train the model based on past experiences
        if len(self.memory) < self.batch_size:
            minibatch = random.sample(self.memory, 1)
            current_batch_size = 1
        else:
            minibatch = random.sample(self.memory, self.batch_size)
            current_batch_size = self.batch_size

        states = np.zeros((current_batch_size, self.state_size))
        targets = np.zeros((current_batch_size, self.action_size))
        td_errors = []

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state
            target = self.model.predict(np.expand_dims(state, axis=0), verbose=0)[0]
            
            if done:
                td_error = reward - target[action]
                target[action] = reward
            else:
                next_q = np.amax(self.model.predict(np.expand_dims(next_state, axis=0), verbose=0)[0])
                td_target = reward + self.gamma * next_q
                td_error = td_target - target[action]
                target[action] = td_target

            td_errors.append(abs(td_error))
            targets[i] = target

        # Perform one step of gradient descent
        self.model.fit(states, targets, epochs=1, verbose=0)

        return np.mean(td_errors)

    def save(self, path):
        # Save model to file
        self.model.save(path)

    def load(self, path):
        # Load model from file
        self.model = tf.keras.models.load_model(path)

# FUNCTIONS
def generate_action_mappings():
    # Generate all possible *permutations* (no repeated moves) of 4 different moves
    moves = ["Up", "Down", "Left", "Right"]
    action_mappings = list(itertools.permutations(moves, 4))  # 24 permutations
    return action_mappings

def natural_sort_key(s):
    # Helper function to naturally sort filenames (ex. 2 before 10)
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

'''
def get_latest_episode_image():
    # Find the most recent episode image from the designated folder
    image_folder = '/content/drive/MyDrive/Deep_Learning_HW_3/episode_images/'
    all_images = glob.glob(image_folder + '*.png')
    all_images = sorted(all_images, key=natural_sort_key)

    if len(all_images) == 0:
        raise ValueError("No images found in episode_images folder!")

    return all_images[-1]  # Return the most recently saved image

def predict_mixing_probability():
    # Predict mixing probability using the latest captured image
    image_path = get_latest_episode_image()
    print(f"Using image for prediction: {image_path}")

    # Load and preprocess the image
    img_raw = tf.io.read_file(image_path)
    img_decoded = tf.image.decode_png(img_raw, channels=3)
    img_resized = tf.image.resize(img_decoded, (64, 64))  # Match CNN input size
    img_normalized = img_resized / 255.0
    img_expanded = tf.expand_dims(img_normalized, axis=0)  # Add batch dimension

    # Predict probability using the loaded CNN model
    prob = cnn_model.predict(img_expanded, verbose=0)
    return prob[0][0]  # Return the predicted probability (single float)

'''

class Simulation():
    def __init__(self):
        self.directions = ['Up','Down','Left','Right']
        self.object_shapes_handles=[]
        self.obj_type = "Cylinder"
        self.initializeSim()

    def initializeSim(self):
        self.client = RemoteAPIClient()
        self.client.setStepping(True)
        self.sim = self.client.getObject('sim')
        self.defaultIdleFps = self.sim.getInt32Param(self.sim.intparam_idle_fps)
        self.sim.setInt32Param(self.sim.intparam_idle_fps, 0) 

        self.sim.loadScene('/Users/alexis/Desktop/Deep_Learning/Project_1_Submission/mix_intro_AI_final.ttt')
        
        self.getObjectHandles()
        self.sim.startSimulation()
        self.dropObjects()
    
    def getObjectHandles(self):
        self.tableHandle=self.sim.getObject('/Table')
        self.boxHandle=self.sim.getObject('/Table/Box')

        self.sim.setEngineFloatParam(self.sim.bullet_body_friction, self.boxHandle, 0.06)
    
    def dropObjects(self):
        topBottom = 0
        
        self.blocks = 8
        frictionCube = 0.06
        blockLength = 0.016
        massOfBlock = 14.375e-03

        self.client.step()
        base_position = self.sim.getObjectPosition(self.boxHandle)
        delta = 0.016
        blue_color = [0.0, 0.0, 1.0]
        red_color = [1.0, 0.0, 0.0]

        for i in range(-2,2):
            new_position = base_position.copy()
            if topBottom:
                new_position[1] += 2*delta # Original, up/down
                new_position[0] += (i + 1) * delta # Original, up/down
            else:
                new_position[0] += 2*delta #left/right
                new_position[1] += (i + 1) * delta #left/right

            new_position[2] += delta
            
            blue_handle = self.sim.createPrimitiveShape(self.sim.primitiveshape_cylinder, [blockLength, blockLength, blockLength], 0)
            self.sim.setObjectPosition(blue_handle, -1, new_position)
            self.sim.setShapeColor(blue_handle, None, self.sim.colorcomponent_ambient_diffuse, blue_color)
            self.sim.setEngineFloatParam(self.sim.bullet_body_friction, blue_handle, frictionCube)
            self.sim.setObjectFloatParam(blue_handle, self.sim.shapefloatparam_mass, massOfBlock)
            self.sim.setObjectInt32Param(blue_handle, self.sim.shapeintparam_static, 0)
            self.sim.setObjectInt32Param(blue_handle, self.sim.shapeintparam_respondable, 1)
            self.sim.resetDynamicObject(blue_handle)

        for i in range(-2,2):
            new_position = base_position.copy()
            if topBottom:
                new_position[1] -= 2*delta # Original, up/down
                new_position[0] += (i + 1) * delta # Original, up/down
            else:
                new_position[0] -= 2*delta #left/right
                new_position[1] += (i + 1) * delta #left/right      

            new_position[2] += delta

            red_handle = self.sim.createPrimitiveShape(self.sim.primitiveshape_cylinder, [blockLength, blockLength, blockLength], 0)
            self.sim.setObjectPosition(red_handle, -1, new_position)
            self.sim.setShapeColor(red_handle, None, self.sim.colorcomponent_ambient_diffuse, red_color)
            self.sim.setEngineFloatParam(self.sim.bullet_body_friction, red_handle, frictionCube)
            self.sim.setObjectFloatParam(red_handle, self.sim.shapefloatparam_mass, massOfBlock)
            self.sim.setObjectInt32Param(red_handle, self.sim.shapeintparam_static, 0)
            self.sim.setObjectInt32Param(red_handle, self.sim.shapeintparam_respondable, 1)
            self.sim.resetDynamicObject(red_handle)
    
    def action(self,shakes, span, direction=None):
        if direction not in self.directions:
            print(f'Direction: {direction} invalid, please choose one from {self.directions}')
            return
        box_position = self.sim.getObjectPosition(self.boxHandle,self.sim.handle_world)
        _box_position = box_position
        #span = 0.02 # Original
        if direction == 'Up':
            idx = 1
            dirs = [1, -1]
        elif direction == 'Down':
            idx = 1
            dirs = [-1, 1]
        elif direction == 'Right':
            idx = 0
            dirs = [1, -1]
        elif direction == 'Left':
            idx = 0
            dirs = [-1, 1]

        for _dir in dirs:
            for _ in range(shakes):
                _box_position[idx] += _dir*span
                #_box_position[idx] += _dir*span / shakes # Original
                self.sim.setObjectPosition(self.boxHandle, self.sim.handle_world, _box_position)
                self.stepSim()

    def stepSim(self):
        self.client.step()

    def stopSim(self):
        self.sim.stopSimulation()

def predict_from_image(img, model):
    img_resized = cv2.resize(img, (64, 64))
    img_normalized = img_resized / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)
    prob = model.predict(img_expanded, verbose=0)
    return prob[0][0]

def capture_image(env, vision_sensor):
    env.sim.handleVisionSensor(vision_sensor)
    img_bytes, resolution = env.sim.getVisionSensorImg(vision_sensor)
    img = np.frombuffer(img_bytes, dtype=np.uint8).reshape((resolution[1], resolution[0], 3))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.flip(img, 1)  # horizontal flip
    img = cv2.rotate(img, cv2.ROTATE_180)  # 180 degree rotation
    img = cv2.resize(img, (64, 64))
    return img

def main():
    # Load the trained CNN model
    global cnn_model
    cnn_model = tf.keras.models.load_model('/Users/alexis/Desktop/Deep_Learning/HW_3/my_model_CNN_best_model_HW3.keras', compile=False)

    # Initialize DQN Agent
    action_mappings = generate_action_mappings()
    agent = DQNAgent(state_size=1, action_size=len(action_mappings))

    # Clear the training log at the start
    log_file_path = "/Users/alexis/Desktop/Deep_Learning/HW_3/training_log.txt"
    log_file = open(log_file_path, "w")
    log_file.write("")  # Overwrite/clear
    log_file.close()

    episodes = 150
    steps_per_episode = 100

    for episode in range(episodes):
        print(f"\n=== Starting Episode {episode + 1}/{episodes} ===")
        
        # Append to log file
        log_file = open(log_file_path, "a")
        log_file.write(f"\n=== Starting Episode {episode + 1}/{episodes} ===\n")

        env = Simulation()
        vision_sensor = env.sim.getObject("/Box/visionSensor")

        img = capture_image(env, vision_sensor)
        previous_prob = predict_from_image(img, cnn_model)

        step = 0
        done = False
        total_reward = 0

        while not done and step < steps_per_episode:
            # Select action
            action_index = agent.act(np.array([previous_prob]))
            directions = action_mappings[action_index]

            # Execute movement
            shakes = 4
            span = 0.003 #0.003
            for direction in directions:
                env.action(shakes=shakes, span=span, direction=direction)

            # Capture New Image
            img = capture_image(env, vision_sensor)

            # Predict new probability
            current_prob = predict_from_image(img, cnn_model)

            # Calculate reward
            if current_prob > 0.8 and previous_prob > 0.8:
                reward = 500
                done = True
            elif current_prob > 0.8:
                reward = 0
            else:
                reward = -5  # fixed penalty

            # Store experience and learn
            agent.remember(np.array([previous_prob]), action_index, reward, np.array([current_prob]), done)
            td_error = agent.replay()

            # Print and Log per step
            if td_error is not None:
                step_message = f"Step {step}: Predicted Mixing: {current_prob:.3f}, Reward: {reward:.2f}, TD Error: {td_error:.4f}"
            else:
                step_message = f"Step {step}: Predicted Mixing: {current_prob:.3f}, Reward: {reward:.2f}"

            print(step_message)
            log_file.write(step_message + "\n")

            # Update
            total_reward += reward
            previous_prob = current_prob
            step += 1

        # Decay epsilon after each episode
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        # End of episode
        episode_summary = f"=== Episode {episode + 1} complete! ===\nAccumulated Reward: {total_reward}\nCurrent epsilon: {agent.epsilon:.4f}\n"
        print(episode_summary)
        log_file.write(episode_summary + "\n")

        # Cleanup
        env.stopSim()
        while env.sim.getSimulationState() != env.sim.simulation_stopped:
            pass

        log_file.close()

    agent.save('/Users/alexis/Desktop/Deep_Learning/HW_3/dqn_agent_trained.keras')
    print("Training Finished!")

if __name__ == '__main__':
    
    main()
