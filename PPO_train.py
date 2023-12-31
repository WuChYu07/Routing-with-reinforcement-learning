import numpy as np
import torch
from Net import ACNetwork
from routing_gym import RoutingEnv
from IL_expert_alternate import Expert, generate_coordinates, switcher, RoutePainter
from PPO_structure import PPO


if __name__ == "__main__":
    expert = Expert()
    num_agents = 5

    start_pos, end_pos = generate_coordinates()
    env = RoutingEnv(start_pos, end_pos)
    max_step = env.wsize ** 2

    device = torch.device('cuda')
    PPO_model = ACNetwork().to(device)
    #PPO_model.load_state_dict(torch.load('RL_agent_a.pt'))

    # ----------------------------------------- #
    # 參數設置
    # ----------------------------------------- #

    #num_episodes = 99999999  # 迭代次數
    actor_lr = 2e-6  # 策略網路學習率
    critic_lr = 2e-6  # 價值網路學習率
    lmbda = 0.95  # 優勢函數系數
    epochs = 10  # 一個批次訓練的次數
    eps = 0.2  # PPO中限制更新範圍參數
    gamma = 0.95  # 折扣因子

    # ----------------------------------------- #
    # model
    # ----------------------------------------- #

    centralized_agent = PPO(0, PPO_model, actor_lr, critic_lr, lmbda, epochs, eps, gamma,
                            device)
    PPO_agent = [PPO(i + 1, PPO_model, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device) for i in
                 range(num_agents)]

    # ----------------------------------------- #
    # 訓練--回合更新 on_policy
    # ----------------------------------------- #

    # done = False
    # start_pos, end_pos = generate_coordinates()
    # start_pos, end_pos, shorts = expert.instruct(start_pos, end_pos)
    # while shorts:
    #     start_pos, end_pos = generate_coordinates()
    #     start_pos, end_pos, shorts = expert.instruct(start_pos, end_pos)

    i = 0
    while True:
        done = False
        start_pos, end_pos = generate_coordinates()
        start_pos, end_pos, shorts = expert.instruct(start_pos, end_pos)
        while shorts:
            start_pos, end_pos = generate_coordinates()
            start_pos, end_pos, shorts = expert.instruct(start_pos, end_pos)

        state = env.reset(start_pos, end_pos)  # 環境重製
        episode_return, executing_id, num_step = 0, 0, 0  # 總reward, 正在執行動作的agent ID
        # 儲存每個episode的數據
        transition_dict = {
            'observations': [],
            'distances': [],
            'actions': [],
            'positions': [],
            'next_observations': [],
            'next_distances': [],
            'next_positions': [],
            'rewards': [],
            'dones': [],
        }
        while not done and num_step < max_step:
            executing_id = switcher(executing_id, num_agents, state)
            if i % 100 < 1 :
                action = PPO_agent[executing_id-1].stochastic_action(state)
            else:
                action = expert.policy[num_step]
            next_state, reward, done, _ = env.step(executing_id, action)  # 环境更新
            transition_dict['observations'].append(state.getObservation(executing_id))
            transition_dict['distances'].append(state.getDistance(executing_id))
            transition_dict['positions'].append(state.getPos(executing_id))
            transition_dict['actions'].append(action)
            transition_dict['next_observations'].append(next_state.getObservation(executing_id))
            transition_dict['next_distances'].append(next_state.getDistance(executing_id))
            transition_dict['next_positions'].append(next_state.getPos(executing_id))
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            # 更新狀態
            state = next_state
            num_step += 1
            # 累績獎勵
            episode_return += reward



        # 模型訓練
        for key in transition_dict:
            transition_dict[key] = np.array(transition_dict[key])

        centralized_agent.learn(transition_dict)
        if done :
            name = 'routing_pic' + str(i) + '.png'
            RoutePainter(20, state.paths, name)


        print('Episode', i, ':', episode_return, 'Done: ', done)
        if i % 100 == 0 and i != 0:
            torch.save(PPO_model.state_dict(), 'RL_agent_a.pt')

        i += 1
