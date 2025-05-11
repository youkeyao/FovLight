## train
```bash
# create new session
tmux new -s FovLight
torchrun --nproc_per_node 2 train.py
# exist Ctrl + b -> d
# connect to session
tmux attach -t FovLight
# kill session
tmux kill-session -t FovLight
```