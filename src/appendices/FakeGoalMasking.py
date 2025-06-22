import numpy as np
import matplotlib.pyplot as plt

# Grid setup
GRID_SIZE = 5

# Define the starting positions for Agent A and Agent B.
A_start = np.array([0, 0])
B_start = np.array([4, 0])

# Define the real and fake goals for Agent A.
A_goal_real = np.array([4, 4])
A_goal_fake = np.array([0, 4])

# Define when Agent A switches its true goal from fake to real.
switch_step = 5 

# Define the total number of simulation steps.
steps = 10

# Simple planner: move toward a goal
def move_toward(pos, goal):
    direction = goal - pos
    step = np.clip(direction, -1, 1)
    return pos + step
    
# Belief update: based on distance
def update_belief(pos, fake, real):
    d_fake = np.linalg.norm(fake - pos)
    d_real = np.linalg.norm(real - pos)
    p_fake = 1 / (d_fake + 1e-3)
    p_real = 1 / (d_real + 1e-3)
    total = p_fake + p_real
    return p_fake / total, p_real / total
    
# Simulate
A_pos = A_start.copy()
B_pos = B_start.copy()
A_path = [A_pos.copy()]
B_path = [B_pos.copy()]
beliefs = []
for t in range(steps):
    goal = A_goal_fake if t < switch_step else A_goal_real
    A_pos = move_toward(A_pos, goal)
    B_belief = update_belief(A_pos, A_goal_fake, A_goal_real)
    if B_belief[1] > 0.5:  # If B thinks A goes to real goal
        B_pos = move_toward(B_pos, A_goal_real)
    else:
        B_pos = move_toward(B_pos, A_goal_fake)
    A_path.append(A_pos.copy())
    B_path.append(B_pos.copy())
    beliefs.append(B_belief)
    
# Plot
A_path = np.array(A_path)
B_path = np.array(B_path)
beliefs = np.array(beliefs)

# Visualize the simulation results: agent paths and belief dynamics.
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(A_path[:, 0], A_path[:, 1], '-o', label='Agent A')
plt.plot(B_path[:, 0], B_path[:, 1], '-s', label='Agent B')
plt.scatter(*A_goal_real, c='red', label='Real Goal', s=100)
plt.scatter(*A_goal_fake, c='orange', label='Fake Goal', s=100)
plt.title('Paths of Agent A (deceptive) and B (predictive)')
plt.legend()
plt.grid(True)

# Subplot 2: Agent B Belief Update Over Time
plt.subplot(1, 2, 2)
plt.plot(beliefs[:, 0], label='B: Belief in Fake Goal')
plt.plot(beliefs[:, 1], label='B: Belief in Real Goal')
plt.axvline(switch_step, color='gray', linestyle='--', label='Switch Point')
plt.title('Agent B Belief Update Over Time')
plt.xlabel('Time Step')
plt.ylabel('Belief')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show() 

