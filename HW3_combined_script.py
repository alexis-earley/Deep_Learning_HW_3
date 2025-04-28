# ==========================================
# Imports
# ==========================================
import sys
sys.path.append('/Applications/coppeliaSim.app/Contents/Resources/programming/zmqRemoteApi/clients/python/src')
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
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# ==========================================
# CODE SECTION 1: DQN Agent Class
# ==========================================
# This section defines the Deep Q-Learning agent.
# It handles storing experiences (state, action, reward, next state), picking actions using epsilon-greedy policy,
# training the neural network (Q-network) based on experience replay,
# and saving/loading the trained model.
# The DQN model predicts Q-values for all possible actions given the current state.
# ==========================================
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.epsilon = 0.5
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.95
        self.learning_rate = 0.001
        self.batch_size = 32
        self.memory = deque(maxlen=50000)
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_size,)),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
        return np.argmax(q_values[0])

    def replay(self):
        # If not enough samples, sample 1; else sample a full batch
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

            td_errors.append(abs(td_error))  # store absolute TD error
            targets[i] = target

        self.model.fit(states, targets, epochs=1, verbose=0)

        return np.mean(td_errors)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)

# ==========================================
# CODE SECTION 2: Generate Action Mappings
# ==========================================
# This section generates all possible 4-move shaking sequences.
# It uses all 24 permutations of the directions ('Left', 'Right', 'Up', 'Down'),
# with each permutation representing one possible action choice the agent can make.

def generate_action_mappings():
    directions = ['Left', 'Right', 'Up', 'Down']
    return list(permutations(directions, 4))

# ==========================================
# CODE SECTION 3: Utility Functions
# ==========================================
# This section defines utility functions used throughout training:
# - `natural_sort_key` ensures filenames are sorted properly even with numbers.
# - `get_latest_episode_image` finds the most recent uploaded screenshot.
# - `predict_mixing_probability` loads the latest image, preprocesses it, and uses the CNN to predict how mixed the object is.

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def get_latest_episode_image():
    image_folder = '/content/drive/MyDrive/Deep_Learning_HW_3/episode_images/'
    all_images = glob.glob(image_folder + '*.png')
    all_images = sorted(all_images, key=natural_sort_key)
    if len(all_images) == 0:
        raise ValueError("No images found in episode_images folder!")
    return all_images[-1]

def predict_mixing_probability():
    image_path = get_latest_episode_image()
    print(f"Using image for prediction: {image_path}")  # <<< ADD THIS LINE
    img_raw = tf.io.read_file(image_path)
    img_decoded = tf.image.decode_png(img_raw, channels=3)
    img_resized = tf.image.resize(img_decoded, (64, 64))
    img_normalized = img_resized / 255.0
    img_expanded = tf.expand_dims(img_normalized, axis=0)
    prob = cnn_model.predict(img_expanded, verbose=0)
    return prob[0][0]

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
        
        # When simulation is not running, ZMQ message handling could be a bit
        # slow, since the idle loop runs at 8 Hz by default. So let's make
        # sure that the idle loop runs at full speed for this program:
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

        print('Blocks finished dropping')
    
    def action(self,shakes=5, span = 0.02, direction=None):
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


    '''
    def capture_image(env, vision_sensor):
        env.sim.handleVisionSensor(vision_sensor)
        img_bytes, resolution = env.sim.getVisionSensorImg(vision_sensor)
        color_image = np.frombuffer(img_bytes, np.uint8)
        color_image = color_image.reshape((resolution[1], resolution[0], 3))
        color_image = cv2.flip(color_image, 1)
        color_image = cv2.rotate(color_image, cv2.ROTATE_180)
        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        resized_image = cv2.resize(color_image, (64, 64))
        return resized_image

    def save_image(img, episode, step):
        filename = f'/content/drive/MyDrive/Deep_Learning_HW_3/episode_images/img_ep{episode+1}_step{step}.png'
        cv2.imwrite(filename, img)
    '''
def predict_from_image(img):
    img_normalized = img / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)
    prob = cnn_model.predict(img_expanded, verbose=0)
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
    # ======== SETUP ========
    # Load the trained CNN model
    global cnn_model
    cnn_model = tf.keras.models.load_model('/Users/alexis/Desktop/Deep_Learning/HW_3/my_model_CNN_best_model_HW3.keras', compile=False)

    # Initialize DQN Agent
    agent = DQNAgent(state_size=1, action_size=24)
    action_mappings = generate_action_mappings()

    # Clear the training log at the start
    log_file_path = "/Users/alexis/Desktop/Deep_Learning/HW_3/training_log.txt"
    log_file = open(log_file_path, "w")
    log_file.write("")  # Overwrite/clear
    log_file.close()

    # ======== TRAINING LOOP ========
    episodes = 100
    steps_per_episode = 100

    for episode in range(episodes):
        print(f"\n=== Starting Episode {episode + 1}/{episodes} ===")
        
        # Append to log file
        log_file = open(log_file_path, "a")
        log_file.write(f"\n=== Starting Episode {episode + 1}/{episodes} ===\n")

        env = Simulation()
        vision_sensor = env.sim.getObject("/Box/visionSensor")

        img = capture_image(env, vision_sensor)
        previous_prob = predict_from_image(img)

        step = 0
        done = False
        total_reward = 0

        while not done and step < steps_per_episode:
            # 1. Select action
            action_index = agent.act(np.array([previous_prob]))
            directions = action_mappings[action_index]

            # 2. Execute movement
            shakes = 4
            span = 0.003
            for direction in directions:
                env.action(shakes=shakes, span=span, direction=direction)

            # 3. Capture New Image
            img = capture_image(env, vision_sensor)

            # 4. Predict new probability
            current_prob = predict_from_image(img)

            # 5. Calculate reward
            if current_prob > 0.8 and previous_prob > 0.8:
                reward = 500
                done = True
            elif current_prob > 0.8:
                reward = 0
            else:
                reward = -5  # fixed penalty

            # 6. Store experience and learn
            agent.remember(np.array([previous_prob]), action_index, reward, np.array([current_prob]), done)
            td_error = agent.replay()

            # 7. Print and Log per step
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

    # ======== FINISH ========
    agent.save('/Users/alexis/Desktop/Deep_Learning/HW_3/dqn_agent_trained.keras')
    print("Training Finished!")

if __name__ == '__main__':
    
    main()
