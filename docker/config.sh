# Config for running project with docker

# Name of image: default = <username>/exposed
image=${EXPA_IMAGE:-"${USER}/exposed-mc"}
# Gpus to run with
gpus=${EXPA_GPUS:-all}
cuda_devices="0"
# Wandb login (and maybe other, this is quite promiscious)
wandb_key=${EXPA_WANBD_KEY:-""}
# Wandb db mode (WANDB_MODE in [run, offline, dryrun, online, disabled])
wandb_mode=${EXPA_WANDB_MODE:-"run"}
# Jupyter notebook port
nb_port=${EXPA_NB_PORT:-"12345"}
# Whether to execute container as root
as_root=${EXPA_AS_ROOT:-"1"}
