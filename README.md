# exposed-mc
Code for the paper "Vessel-to-Vessel Motion Compensation with Reinforcement Learning

## Setup

Below follows instructions for setting up the codebase.

### Docker 

This project is constructed with docker in mind. However, if you don't want to use docker, you can simply install the dependencies in `docker/requirements.txt`, along with pytorch, in a normal python environment.

#### Building

To build the docker iamge run: 

```bash
./docker/build.sh
```
This will build a docker image with the name `<username>/exposed-mc` where `<username>` is the name of the current logged in user.


#### Running 

To run any python command:
```
./docker/run.sh python <script-name> args...
```
This will start a container of the image we just built with the current directory mounted in with live code.
By default, it will start the container in interactive mode with your terminal attached.
If you want to run in detached mode, set the environment variable `export EXPA_DOCKER_ARG="-d"` first.

#### Configuring 

The docker setup has a config file `docker/config.sh` with a handful of environment variables you might want to tune. 
For instance, you can configure which GPU to pass into the container. 
Most of these variables can also be set from the shell through environment variables.

### Weights and Biases

Experiment tracking for training is done with weights and biases. You don't need to use it, but it is recommended for keeping track of misc metrics and diagnositcs during training. 
If you haven't already, create a free account on [wandb.ai](wandb.ai). 
Afterwards, navigate to `Settings` (top right menu), scroll down to `Danger Zone` and create an API key.
Copy the API key an set it in the environment variable `export EXPA_WANBD_KEY=<your-api-key>`.

## Training 

To train a PPO RL agent for an environment, execute:

```bash
docker/run.sh python -m exposed.agents.ppo training/ --seed <seed> --wandb <wandb-project> --name "<run-name>" -c.sampling.env_name <env-name>
```

This will grab the PPO config from the `./training/config.yml` folder and launch training with seed `<seed>` and environment `<env-name>`.
The `--wandb` argument tells the program which Weights and Biases project to file the run under (add a colon to sub-specify run group). 
The `--name` argument tells the program gives the run a name, both as a sub-folder of `./training`, and in wandb.
The `-c.sampling.env_name` argument overrides the config with the environment used in this run. The naming convention for training envs is:

```
v2v:train_<delay-filter>
```
Where delay filter can be `delay_0`, `delay_20`, or `delay_40` (explicit delay), or `smooth_20`, `smooth_40` (butterworth delays).

For instance, to train the model for the first seed of delay configuration $\mathcal{D}_B(4, 21)$, we do:

```bash
docker/run.sh python -m exposed.agents.ppo training/ --seed 0 --wandb exposed-mcx:smooth_20 --name "mcx_smooth_20-$SEED" -c.sampling.env_name v2v:train_smooth_20
```

## Evaluating 

To evaluate the agent we just trained, execute:

```bash
docker/run.sh python evaluate.py v2v:eval_hs25_smooth_20 ppo --greedy --seed 0 --episodes 50 --exp training/mcx_smooth_20-0 --save results_ppo.h5
```
This will evaluate the trained agent (with config and network parameters found in `training/mcx_smooth_20-0`) on an environment with delay configuration $\mathcal{D}_B(4, 21)$ and sea state $\mathcal{W}_d$ (significant wave height = 2.5 meter) for 50 episodes and save all observations, actions, and rewards in a HDF file called `results_ppo.h5`

To run the baseline agent on the same environment, execute:
```bash
docker/run.sh python evaluate.py v2v:eval_hs10_delay_0 constant --seed 0 --episodes 50 --save results_baseline.h5
```

This will generate corresponding results for the baseline. Note that since we use environment seed `0` for both evaluations, the vessel motions encountered in both runs will be exactly the same.

