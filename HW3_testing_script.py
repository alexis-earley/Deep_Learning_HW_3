import numpy as np
import tensorflow as tf
import random
from HW3_training_script import Simulation, capture_image, predict_from_image, generate_action_mappings, DQNAgent

# ====== Setup ======
# Load CNN model and DQN agent
cnn_model = tf.keras.models.load_model('/Users/alexis/Desktop/Deep_Learning/HW_3/my_model_CNN_best_model_HW3.keras', compile=False)
agent = DQNAgent(state_size=1, action_size=24)
agent.model = tf.keras.models.load_model('/Users/alexis/Desktop/Deep_Learning/HW_3/dqn_agent_trained.keras', compile=False)
action_mappings = generate_action_mappings()

# Paths
log_path = '/Users/alexis/Desktop/Deep_Learning/HW_3/testing_log.txt'

# ====== Parameters ======
num_trials = 100
max_steps = 10  # 10 shakes per trial

# ====== Helper function ======
def run_trial(use_agent=True):
    env = Simulation()
    vision_sensor = env.sim.getObject("/Box/visionSensor")
    img = capture_image(env, vision_sensor)
    previous_prob = predict_from_image(img, cnn_model)

    success = False
    for step in range(max_steps):
        if use_agent:
            action_index = agent.act(np.array([previous_prob]))
        else:
            action_index = random.randint(0, 23)

        directions = action_mappings[action_index]

        # Perform action
        shakes = 4
        span = 0.003
        for direction in directions:
            env.action(shakes=shakes, span=span, direction=direction)

        # Capture new image and predict
        img = capture_image(env, vision_sensor)
        current_prob = predict_from_image(img, cnn_model)

        # --- Fix: success only if both previous and current > 0.8 ---
        if previous_prob > 0.8 and current_prob > 0.8:
            success = True
            break

        previous_prob = current_prob  # Update for next comparison

    env.stopSim()
    while env.sim.getSimulationState() != env.sim.simulation_stopped:
        pass

    return success, step + 1  # Return whether mixed and how many steps used

# ====== Run tests ======
with open(log_path, "w") as f:
    # Test using trained agent
    f.write("=== Testing Trained Agent ===\n")
    print("\n=== Testing Trained Agent ===")
    successes = 0
    for trial in range(num_trials):
        success, steps_used = run_trial(use_agent=True)
        if success:
            successes += 1
        f.write(f"Trial {trial+1}: {'Success' if success else 'Failure'} in {steps_used} steps\n")
        print(f"Trial {trial+1}: {'Success' if success else 'Failure'} in {steps_used} steps")
    f.write(f"Total Success Rate: {successes}/{num_trials}\n\n")
    print(f"Total Success Rate: {successes}/{num_trials}")

    # Test using random actions
    f.write("=== Testing Random Actions ===\n")
    print("\n=== Testing Random Actions ===")
    successes = 0
    for trial in range(num_trials):
        success, steps_used = run_trial(use_agent=False)
        if success:
            successes += 1
        f.write(f"Trial {trial+1}: {'Success' if success else 'Failure'} in {steps_used} steps\n")
        print(f"Trial {trial+1}: {'Success' if success else 'Failure'} in {steps_used} steps")
    f.write(f"Total Success Rate: {successes}/{num_trials}\n\n")
    print(f"Total Success Rate: {successes}/{num_trials}")

print(f"Testing complete. Results saved to '{log_path}'.")
