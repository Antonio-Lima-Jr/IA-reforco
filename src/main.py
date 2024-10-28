from IA.dqn_model import DQN
from IA.dqn_agent import DQNAgent
#  importar degrees da lib math
from torch.utils.tensorboard import SummaryWriter

from enviroment.cartpole import MyCartPoleEnv


EPISODES = 1000
LEARNING_RATE = 0.001
input_size = 4
output_size = 2
DQN_model = DQN(input_size, output_size)
env = MyCartPoleEnv(model=DQN_model, render_mode="human") 
# epsilon = 1.0
# epsilon_min = 0.01
# epsilon_decay = 0.995

DQN_agent = DQNAgent(DQN_model)


for episode in range(EPISODES):
    writer = SummaryWriter()
    state, _ = env.reset()
    total_reward = 0
    done = False
    exit = False
    cont_t = 0
    parans_model = None

    while not exit:

        action = DQN_agent.select_action(state, output_size)

        next_state, reward, done, info, dict = env.step(action, parans_model)
        total_reward += DQN_agent.remember(state, action, reward, next_state, done)
        parans_model = DQN_agent.train_model()
  
        if done:
            exit = done
            print(f"Episódio terminou após {cont_t+1} passos")
            break
        state = next_state
        cont_t += 1
    
    DQN_agent.update_epsilon()

    writer.add_scalar("Loss/train", cont_t, episode)

    print(f"Episódio {episode + 1}/{EPISODES}, Recompensa Total: {total_reward}")
    writer.flush()

env.close()
