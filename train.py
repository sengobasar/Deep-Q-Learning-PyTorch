import gymnasium as gym
import numpy as np
import torch

from agent import DQNAgent


# --------------------------------------------------
# Training Function
# --------------------------------------------------

def train_dqn(
    env_name="CartPole-v1",
    episodes=500,
    render=False,
    save_path="dqn_model.pth"
):

    # Create environment
    env = gym.make(env_name, render_mode="human" if render else None)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print("Environment:", env_name)
    print("State dim:", state_dim)
    print("Action dim:", action_dim)

    # Create agent
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        use_dueling=False   # Change to True for Dueling DQN
    )

    rewards_history = []
    losses_history = []


    # --------------------------------------------------
    # Training Loop
    # --------------------------------------------------

    for episode in range(1, episodes + 1):

        state, _ = env.reset()
        total_reward = 0
        done = False


        while not done:

            # Choose action
            action = agent.select_action(state)

            # Step environment
            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            # Store transition
            agent.store(
                state,
                action,
                reward,
                next_state,
                done
            )

            # Train
            loss = agent.train_step()

            if loss is not None:
                losses_history.append(loss)

            state = next_state
            total_reward += reward


        rewards_history.append(total_reward)


        # --------------------------------------------------
        # Logging
        # --------------------------------------------------

        if episode % 10 == 0:

            avg_reward = np.mean(rewards_history[-10:])
            avg_loss = (
                np.mean(losses_history[-50:])
                if len(losses_history) > 0
                else 0
            )

            print(
                f"Episode {episode:4d} | "
                f"Avg Reward: {avg_reward:6.2f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Loss: {avg_loss:.4f}"
            )


    # --------------------------------------------------
    # Save Model
    # --------------------------------------------------

    agent.save(save_path)
    print("\nModel saved to:", save_path)

    env.close()

    return agent, rewards_history



# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    agent, rewards = train_dqn(
        env_name="LunarLander-v3",
        episodes=500,
        render=True
    )
