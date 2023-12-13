import torch
from Net import ACNetwork
from routing_gym import RoutingEnv
from IL_expert_alternate import Expert, generate_coordinates, RoutePainter, switcher
from PPO_structure import PPO




if __name__ == "__main__":
    e = Expert()
    num_agents = 5

    start_pos, end_pos = [(13, 17), (9, 19), (19, 8), (13, 13), (6, 1)], [(1, 11), (1, 5), (15, 16), (19, 12), (4, 3)]
    env = RoutingEnv(start_pos, end_pos)
    max_step = env.wsize ** 2

    device = torch.device('cuda')
    PPO_model = ACNetwork().to(device)
    PPO_model.load_state_dict(torch.load('RL_agent_a.pt'))

    # ----------------------------------------- #
    # 參數設置
    # ----------------------------------------- #

    num_episodes = 100000  # 迭代次數
    actor_lr = 2e-5  # 策略網路學習率
    critic_lr = 2e-5  # 價值網路學習率
    lmbda = 0.95  # 優勢函數系數
    epochs = 10  # 一個批次訓練的次數
    eps = 0.2  # PPO中限制更新範圍參數
    gamma = 0.95  # 折扣因子

    # ----------------------------------------- #
    # model
    # ----------------------------------------- #

    PPO_agent = [PPO(i + 1, PPO_model, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device) for i in range(num_agents)]  # decentralized execution

    # ----------------------------------------- #
    # 訓練--回合更新 on_policy
    # ----------------------------------------- #

    done = False
    i = 0
    while not done:
        start_pos, end_pos = generate_coordinates(size=20, n=5)
        if True:
            start_pos, end_pos, shorts = e.instruct(start_pos, end_pos)
            while shorts:
                start_pos, end_pos = generate_coordinates()
                start_pos, end_pos, shorts = e.instruct(start_pos, end_pos)
        state = env.reset(start_pos, end_pos)  # 環境重置
        done = False  # 任務完成
        episode_return, executing_id, num_step = 0, 0, 0  # 總reward, 正在執行動作的agent ID

        print(start_pos, end_pos)

        while not done and num_step < max_step:
            executing_id = switcher(executing_id, num_agents, state)
            action = PPO_agent[executing_id-1].deterministic_action(state, e.policy[num_step])  # 動作選擇
            next_state, reward, done, _ = env.step(executing_id, action)  # 環境更新
            state = next_state
            num_step += 1
            episode_return += reward

        print('Episode', i, 'Total_reward:', episode_return, ', Done: ', done)
        if done:
            RoutePainter(20, state.paths, 'result.png')
        i += 1



