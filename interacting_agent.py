# This file has the actor critic interacting with the environment which should run in parallel
import torch

def interaction_AC(global_model, local_model,local_env, device,
                    T = 0,
                    GAMMA = .9,
                    LAMBDA_G = .9,
                    ENTROPY_COEF = .01,
                    VALUE_LOSS_COEF = 1.4,
                    LEARNING_RATE = 1e-4,
                    WEIGHT_DECAY = None,
                    MAX_GRAD_NORM = 1.5,

                    T_max = 25000 ,
                    # T_max = 3
                    t_sim_max = 200,
                    # t_sim_max = 1
                    T_SNN_VAL_WINDOW = 5):
    
    model = local_model

    model.train()

    env = local_env

    loss = 0
    time_finish = 0
    T_lst = []
    loss_lst = []
    print('Lets try to learn one iteration')
    while T<T_max:
        model.init_mem() 

        T +=1
        print('Run: ',T, '\t time: ',time_finish)
        T_lst.append(time_finish)
    
        # retrieve parameters of global nets
        model_params  = global_model.state_dict()


        model.load_state_dict(model_params)

        # get initial state
        state, info = env.reset()
        # state = env.set_state(constant_state)

        # set up replay memory

        t_sim = 0
        terminal = False
        reward = 0

        values = []
        log_probs = []
        rewards = []
        entropies = []
        # print('Interacting with environment')
        while not terminal and t_sim < t_sim_max:

            # state to tensor
            state = torch.from_numpy(state).to(device)

            # get network outputs on given state
            value, policy = model(state.unsqueeze(0), device=device)

            # find probabilities of certain actions
            prob = F.softmax(policy, dim=-1)

            logprob = F.log_softmax(policy, dim=-1)

            # calculate entropy
            entropy = -(logprob * prob).sum(1, keepdim=True)
            entropies.append(entropy)

            # choose the action and detach from computational graph
            action = prob.multinomial(num_samples=1).detach() # find max of this and make sure it is not part of optimization

            # get log probs of the actions taken
            log_prob = logprob.gather(1,action)
            
            # perform action
            obs, reward, terminal, _, _ = env.step(int(action.squeeze(0)))

            # !!!!!!!!!!!
            reward = max(min(reward, 1), -1)

            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            
            if not terminal:
                state = obs

            t_sim += 1

        time_finish = t_sim   
    
        R = 0
        if not terminal:
            R, _ = model(torch.from_numpy(state).to(device), device=device)
            R = R.detach()

            
        # save current R (value) for the gneralized advantage DeltaT
        prev_val = R
        values.append(R)
        # on github code, they append this to the transitions, 
        # however, they substract values again to calculate advantage, 
        # does this mean they basically neglect this step for the advantage function?
        # my intuition: advantage tells how much better reward was than expected
        # therefore, you have to substract the value from the reward of the correct step.
        
        # print('Optimizing agents and critics...\n\n')
        policy_loss = 0
        value_loss  = 0
        g = torch.zeros(1,1).to(device)
        len_counter = 0
        # note cycle through memory sequentially as we are appending left
        for i in reversed(range(len(rewards))):
            len_counter+=1

            if len_counter>5:

                R = rewards[i] + GAMMA*R

            # accumulate gradients wrt thread specific theta
                advantage = R - values[i] # note that according to the git they are out of sync by one step 

        #     # this if from the paper: 
        #     # question: doesn't this iterative adding make steps too big? all the gradients are applied but they were all calculated to the same initial net? 
        #     # every step now changes the weights but this gradient is wrt the initial weights



                # alternative training loop
                value_loss = value_loss + 0.5*advantage.pow(2)

                # generalized advantage estimation from: https://arxiv.org/pdf/1506.02438.pdf
                delta_t = rewards[i] + GAMMA* values[i+1] - values[i]
                g = g*GAMMA*LAMBDA_G + delta_t

                policy_loss = policy_loss - log_probs[i]*g.detach() - entropies[i]*ENTROPY_COEF

            
            else:
                value_loss = 0
                policy_loss = 0

            # if (T%25000 == 0)   or T==3:
            #     print('Policy loss: ', policy_loss)
            #     print('Value loss: ', value_loss)
            #     plot_activity(torch.stack(model.spk1_rec),torch.stack(model.spk2_rec),torch.stack(model.spk3_rec))
            #     plot_potentials(torch.stack(model.spk1_rec),torch.stack(model.mem1_rec),torch.stack(model.syn1_rec),torch.stack(model.spk2_rec),torch.stack(model.mem2_rec),torch.stack(model.syn2_rec))
        
            optimizer.zero_grad()

            l2_norm = sum(p.pow(2.0).sum()
                        for p in model.parameters())

            spike_sparsity_loss = torch.sum(torch.stack(model.spk_in_rec)) + torch.sum(torch.stack(model.spk1_rec)) + torch.sum(torch.stack(model.spk2_rec)) + torch.sum(torch.stack(model.spk3_rec))
            # print('spike_sparsity loss: ' + str(spike_sparsity_loss*.00005 + 1/spike_sparsity_loss*100))
            # print('other loss: '+ str((policy_loss + value_loss * VALUE_LOSS_COEF )))
            loss = (policy_loss + value_loss * VALUE_LOSS_COEF ) + spike_sparsity_loss*.00005 + 1/(spike_sparsity_loss+1e-6)*100
            # loss = loss
            loss_lst.append(loss.to('cpu').detach().squeeze(0))
            
            loss = loss/len_counter # normalize?

            loss.backward()
            # print('Backward pass completed!')
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)