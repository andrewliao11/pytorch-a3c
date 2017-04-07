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
from six.moves import queue

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(rank, args, shared_model, loss_master, optimizer=None):
    torch.manual_seed(args.seed + rank)

    env = create_atari_env(args.env_name)
    env.seed(args.seed + rank)

    model = ActorCritic(env.observation_space.shape[0], env.action_space)

    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    model.train()

    state = env.reset()
    state = torch.from_numpy(state)
    done = True

    episode_length = 0
    while True:
        episode_length += 1
        # Sync with the shared model every iteration
        model.load_state_dict(shared_model.state_dict())
        if done:
	    # initialization
            cx = Variable(torch.zeros(1, 256))
            hx = Variable(torch.zeros(1, 256))
        else:
            cx = Variable(cx.data)
            hx = Variable(hx.data)

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(args.num_steps):
            value, logit, (hx, cx) = model(
                (Variable(state.unsqueeze(0)), (hx, cx)))
            prob = F.softmax(logit)
            log_prob = F.log_softmax(logit)
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)

            action = prob.multinomial().data
            log_prob = log_prob.gather(1, Variable(action))

            state, reward, done, _ = env.step(action.numpy())
	    # prevent stuck agents
            done = done or episode_length >= args.max_episode_length
	    # reward shaping
            reward = max(min(reward, 1), -1)

            if done:
                episode_length = 0
                state = env.reset()

            state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((Variable(state.unsqueeze(0)), (hx, cx)))
            R = value.data

        values.append(Variable(R))
        R = Variable(R)
        gae = torch.zeros(1, 1)
	# calculate the rewards from the terminal state
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
	    # convert the data into xxx.data will stop the gradient
            delta_t = rewards[i] + args.gamma * \
                values[i + 1].data - values[i].data
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = -log_probs[i] * Variable(gae) - 0.01 * entropies[i]
	    loss_master.put(policy_loss + 0.5 * value_loss)
	    #loss_master.append(policy_loss + 0.5 * value_loss)

	loss = []
	while len(loss) < args.batch_size:
	    try:
		loss.append(loss_master.get_nowait())
	    except queue.Empty:
		break
	if len(loss) > 0:
	    loss = sum(loss)
	    optimizer.zero_grad()
	    loss.backward(retain_variables=True)
	    torch.nn.utils.clip_grad_norm(model.parameters(), 40)
	    ensure_shared_grads(model, shared_model)
	    optimizer.step()
	
