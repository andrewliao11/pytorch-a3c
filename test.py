import math
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
from envs import create_atari_env
from model import ActorCritic
from torch.autograd import Variable
from torchvision import datasets, transforms
import time
from collections import deque


def setup(args):
    # logging
    log_dir = os.path.join('logs', args.model_name)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_filename = args.env_name+"."+args.model_name+".log"
    f = open(os.path.join(log_dir, log_filename), "w")
    # model saver
    ckpt_dir = os.path.join('ckpt', args.model_name)
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
    ckpt_filename = args.env_name+"."+args.model_name+".pkl"
    return f, os.path.join(ckpt_dir, ckpt_filename)

def test(rank, args, shared_model):

    torch.manual_seed(args.seed + rank)

    env = create_atari_env(args.env_name)
    env.seed(args.seed + rank)

    model = ActorCritic(env.observation_space.shape[0], env.action_space)

    model.eval()

    f, ckpt_path = setup(args)
    state = env.reset()
    state = torch.from_numpy(state)
    reward_sum = 0
    done = True

    start_time = time.time()
    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=100)
    episode_length = 0
    try:
    	while True:
            episode_length += 1
            # Sync with the shared model
            if done:
            	model.load_state_dict(shared_model.state_dict())
            	cx = Variable(torch.zeros(1, 256), volatile=True)
            	hx = Variable(torch.zeros(1, 256), volatile=True)
            else:
            	cx = Variable(cx.data, volatile=True)
            	hx = Variable(hx.data, volatile=True)

            value, logit, (hx, cx) = model(
            	(Variable(state.unsqueeze(0), volatile=True), (hx, cx)))
            prob = F.softmax(logit)
            action = prob.max(1)[1].data.numpy()

            state, reward, done, _ = env.step(action[0, 0])
            done = done or episode_length >= args.max_episode_length
            reward_sum += reward

            # a quick hack to prevent the agent from stucking
            actions.append(action[0, 0])
            if actions.count(actions[0]) == actions.maxlen:
            	done = True

            if done:
	    	info_str = "Time {}, episode reward {}, episode length {}".format(
                	time.strftime("%Hh %Mm %Ss",time.gmtime(time.time() - start_time)),
                	reward_sum, episode_length)
	        print(info_str)
	        f.write(info_str+'\n')
                reward_sum = 0
            	episode_length = 0
            	actions.clear()
            	state = env.reset()
            	time.sleep(60)

            state = torch.from_numpy(state)
    except KeyboardInterrupt:
	f.close()
	torch.save(model.state_dict(), ckpt_path)
