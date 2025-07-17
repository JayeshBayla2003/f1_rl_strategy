# src/environment.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

# --- THE REALISTIC F1 SIMULATOR ---
class F1RaceSimulator:
    """
    This class simulates the core mechanics of an F1 race for a single car,
    including realistic tire degradation and mandatory F1 rules.
    """
    def __init__(self, starting_tire='SOFT'):
        # --- Race Parameters ---
        self.total_laps = 57
        self.pit_stop_time = 22  # seconds

        # --- Tire Degradation Parameters ---
        # Lap time loss per lap for each compound
        self.lap_degradation = {'SOFT': 0.18, 'MEDIUM': 0.12, 'HARD': 0.08}
        # The lap after which a tire 'falls off a cliff' and becomes very slow
        self.tire_cliff_age = {'SOFT': 22, 'MEDIUM': 32, 'HARD': 42}
        # The extra time penalty per lap after hitting the cliff
        self.cliff_penalty = 8  # seconds

        # --- Initial State ---
        self.start_compound = starting_tire
        self.reset()

    def reset(self):
        """Resets the simulation to the start of the race."""
        self.current_lap = 1
        self.current_tire = self.start_compound
        self.tire_age = 0
        
        # This dictionary tracks which compounds have been used in the race.
        self.compounds_used = {'SOFT': False, 'MEDIUM': False, 'HARD': False}
        self.compounds_used[self.current_tire] = True
        
        return self._get_state()

    def _check_rule_violation(self):
        """Checks if the two-compound rule has been violated."""
        # The rule is met if the number of unique compounds used is 2 or more.
        num_compounds_used = sum(self.compounds_used.values())
        return num_compounds_used < 2

    def _get_state(self):
        """Returns the current state of the simulation as a NumPy array."""
        # The state now includes which compounds have been used (as 0 or 1).
        return np.array([
            self.current_lap,
            self.tire_age,
            int(self.compounds_used['SOFT']),
            int(self.compounds_used['MEDIUM']),
            int(self.compounds_used['HARD'])
        ])

    def step(self, action):
        """
        Executes one step in the simulation (i.e., one lap) based on the chosen action.
        """
        # --- Action Mapping ---
        # Action 0: Stay Out
        # Action 1: Pit for SOFT
        # Action 2: Pit for MEDIUM
        # Action 3: Pit for HARD
        
        time_loss = 0
        if action > 0:  # Pit stop
            # Can't pit for the same tire type you are currently on
            if (action == 1 and self.current_tire == 'SOFT') or \
               (action == 2 and self.current_tire == 'MEDIUM') or \
               (action == 3 and self.current_tire == 'HARD'):
                # Heavily penalize trying to pit for the same tire
                return self._get_state(), 1000, self.current_lap > self.total_laps

            time_loss = self.pit_stop_time
            self.tire_age = 0
            if action == 1: self.current_tire = 'SOFT'
            elif action == 2: self.current_tire = 'MEDIUM'
            else: self.current_tire = 'HARD'
            self.compounds_used[self.current_tire] = True
        else:  # Stay out
            self.tire_age += 1

        # --- Lap Time Calculation ---
        base_lap_time = 95  # A fictional base lap time in seconds
        degradation_loss = self.tire_age * self.lap_degradation[self.current_tire]
        
        cliff_loss = 0
        if self.tire_age > self.tire_cliff_age[self.current_tire]:
            cliff_loss = self.cliff_penalty
            
        lap_time = base_lap_time + degradation_loss + time_loss + cliff_loss
        
        self.current_lap += 1
        done = self.current_lap > self.total_laps
        
        return self._get_state(), lap_time, done


# --- THE GYMNASIUM ENVIRONMENT WRAPPER ---
class F1StrategyEnv(gym.Env):
    """
    This class wraps the F1RaceSimulator in a Gymnasium environment,
    which is the standard interface for Reinforcement Learning agents.
    """
    def __init__(self):
        super(F1StrategyEnv, self).__init__()
        self.simulator = F1RaceSimulator()
        
        # Action space: 0=StayOut, 1=PitSoft, 2=PitMedium, 3=PitHard
        self.action_space = spaces.Discrete(4)
        
        # Observation space must now match the new state shape (5 elements).
        # [current_lap, tire_age, soft_used, medium_used, hard_used]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0]), 
            high=np.array([self.simulator.total_laps + 1, self.simulator.total_laps + 1, 1, 1, 1]), 
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        state = self.simulator.reset()
        return state, {}

    def step(self, action):
        """
        Executes a step, calculates the reward, and checks for rule violations.
        """
        next_state, lap_time, done = self.simulator.step(action)
        
        # The base reward is the negative of the lap time.
        # The agent wants to maximize reward, so it will try to minimize lap time.
        reward = -lap_time
        
        # --- Final Reward Adjustment ---
        if done:
            # If the race is over, check if the rules were followed.
            if self.simulator._check_rule_violation():
                # Apply a massive penalty for not using at least two compounds.
                # This teaches the agent that this is an unacceptable outcome.
                reward -= 5000
        
        terminated = done
        truncated = False # Not used in this simple case
        
        return next_state, reward, terminated, truncated, {}