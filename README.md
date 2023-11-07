# SpikingA2C
This repository supports:
https://www.overleaf.com/read/xptjvhsmcxph#3a4180

## File structure:
### main_a2c_alg.py
This file has the main code for running the training and saving the model, training data and plot.

### worker.py
This file contains the worker and global class. Each worker, $i$, interacts with their own environment for a certain amount of time $t_i$. All the workers do this in parallel. They each compute the gradients $\nabla_i$ for the update. The gradients are gathered at the global worker. 
The global gradient $$\nabla_{\text{global}}$$ is calculated as the weighted sum of individual gradients $$\nabla_1, \nabla_2, \ldots, \nabla_n$$ divided by the sum of the weights $$t_1, t_2, \ldots, t_n$$

$$\nabla_{\text{global}} = \frac{t_1 \cdot \nabla_1 + t_2 \cdot \nabla_2 + \ldots + t_n \cdot \nabla_n}{t_1 + t_2 + \ldots + t_n}$$


### actor_critcs.py
The agent models that are used are stored in this file.

### environments.py
This file contains the custom environments used in the experiments. This mainly includes an altered CartPole class, which allows for more customizability for debugging. This includes the option to reset the CartPole to a fixed state, changing the timestep easily,...

### evaluation_pipeline.py
The methods to perform basic evaluation of the models, are presented here. This includes a noise analysis function, the pruning algorihm and the average reward pipeline.

### fastsigmoid_module.py
A custom implementation of the fastsigmoid surrogate function was required to allow for the pickling of the SNN based agents for parallelization.
