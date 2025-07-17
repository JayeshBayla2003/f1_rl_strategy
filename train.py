# train.py (Updated Snippet)
import torch
from src.enviorment import F1StrategyEnv
from src.agent import DQNAgent # This now works perfectly
# model.py is no longer directly needed here, as the agent handles it.

# --- Hyperparameters ---
NUM_EPISODES = 500
TARGET_UPDATE_FREQUENCY = 10 # Update the target network every 10 episodes

def main():
    env = F1StrategyEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    n_actions = env.action_space.n
    state, _ = env.reset()
    n_observations = len(state)

    # Initialize Agent
    agent = DQNAgent(n_observations, n_actions, device) # Simplified initialization

    # --- Training Loop ---
    for i_episode in range(NUM_EPISODES):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        done = False
        while not done:
            action = agent.select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory and optimize
            agent.memory.push(state, action, next_state, reward)
            state = next_state
            agent.optimize_model()

        # Update the target network after a certain number of episodes
        if i_episode % TARGET_UPDATE_FREQUENCY == 0:
            agent.update_target_net()
            print(f"Episode {i_episode}: Target network updated.")

        # Log progress
        if i_episode % 50 == 0:
             print(f"Episode {i_episode} complete.")


    print("Training complete.")
    # Save the trained model's state dictionary
    torch.save(agent.policy_net.state_dict(), 'f1_strategy_dqn.pth')

if __name__ == '__main__':
    main()