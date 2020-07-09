#!/bin/bash

sudo find -name "*__pycache__*" | sudo xargs -I{} rm -r '{}'
sudo find -name "*.ipynb_checkpoints*" | sudo xargs -I{} rm -r '{}'

sudo chown -R 1000 experiments/results
sudo chown -R 1000 experiments/tensorboard_logs
sudo chgrp -R 1000 experiments/results
sudo chgrp -R 1000 experiments/tensorboard_logs