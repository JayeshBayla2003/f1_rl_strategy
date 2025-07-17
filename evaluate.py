# evaluate.py
import torch
from src.enviorment import F1StrategyEnv
from src.model import DQN

def evaluate():
    env = F1StrategyEnv()
    state, _ = env.reset()
    n_observations = len(state)
    n_actions = env.action_space.n

    # Load the trained model
    model = DQN(n_observations, n_actions)
    model.load_state_dict(torch.load('f1_strategy_dqn.pth'))
    model.eval() # Set the model to evaluation mode

    done = False
    total_race_time = 0
    strategy = []

    print("--- Evaluating Optimal Strategy ---")
    while not done:
        with torch.no_grad():
            # Choose the best action, no exploration
            action = model(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).max(1)[1].view(1, 1)
        
        state, reward, terminated, truncated, _ = env.step(action.item())
        total_race_time -= reward # Reward is negative lap time
        done = terminated or truncated

        # Log the decision
        lap = env.simulator.current_lap - 1
        if action.item() > 0:
            tire = ['SOFT', 'MEDIUM', 'HARD'][action.item()-1]
            print(f"Lap {lap}: Pit for {tire} tires.")
            strategy.append((lap, tire))

    print(f"Optimal Strategy Found: {strategy}")
    print(f"Predicted Total Race Time: {total_race_time:.2f} seconds")

if __name__ == '__main__':
    evaluate()