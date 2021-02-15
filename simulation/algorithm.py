import random

from .utils import render_movie
from tqdm import tqdm


def coevolve(board, all_agents, timedraw, resource_max, total_timesteps,
             do_movie=True, do_random_choice=False, log=False,
             img_path="./output/Field_Drive.gif"):

    t = 0
    is_done = False  # This will never be called for now.
    for t in tqdm(range(1, total_timesteps), desc="Running..."):
        # Update the board one-by-one
        for i, agent in enumerate(all_agents):
            state = board.get_observation_from_coords(
                agent.position_x, agent.position_y)
            if do_random_choice:
                action = random.choice([0, 1, 2, 3, 4])
            else:
                action = int(agent.act(state))
            obs, rew, is_done, info = board.step(agent_action=action,
                                                 agent_id=i,
                                                 update_time=(i == len(all_agents) - 1))
            agent.update(state, rew, info["success"])
            if log:
                print("{} > s: {}; a: {} r: {} | obs: {} ".format(
                    t, state, action, rew, obs))

    if do_movie:
        render_movie(timedraw, len(all_agents),
                    board.POSITION_X,
                    board.POSITION_Y,
                    board.ENVIRONMENT_INFLUENCED_HISTORY,
                    board.ENVIRONMENT_SIZE_X,
                    board.ENVIRONMENT_SIZE_Y,
                    board.AGENT_RADIUS,
                    resource_max,
                    do_random_choice,
                    img_path)
