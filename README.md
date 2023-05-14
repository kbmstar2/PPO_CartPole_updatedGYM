# PPO_CartPole_updatedGYM
This is a repository about PPO algorithm with updated GYM. My version of gym is 0.26.1.

base code of this files : https://www.youtube.com/watch?v=hlv79rcHws0 / https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/PPO/torch
Thanks to Phil for easily coding PPO with gym - using CartPole-v1, but there are several problems with new version of gym.

Firstly, env.reset() now return tuple composed of numpy array and dictionary. 
See definition of reset in core.py:

def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:

Then, error occured when observation is converted to tensor by this code:
state = T.tensor([observation], dtype=T.float).to(self.actor.device)

So, I fixed the code in main.py:
observation = env.reset()
observation = observation[0]

This means: when observation is reset, the form of observation converted to Tuple[ObsType], not Tuple[ObsType, dict].

Secondly, an error occured in self.actor and self.critic because of input_dims.
In main.py, input_dims is defined as input_dims=env.observation_space.shape, but shape of observation_space is Tuple[ObsType,dict], so we have to convert to Tuple[ObsType].

so I fixed the code in main.py:
agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs, 
                    input_dims=env.observation_space.shape[0]).
                    
Last, env.step(action) now return five: 
observation(object), reward(float), terminated(bool), truncated(bool), info(dictionary).
Please check the definition of step() in core.py.
In this code, truncated and info is not very important. 
so I fixed the code in main.py:
observation_, reward, done, _ , _ = env.step(action).




