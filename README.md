## dependency
```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
wget https://anaconda.org/pytorch3d/pytorch3d/0.7.8/download/linux-64/pytorch3d-0.7.8-py39_cu118_pyt241.tar.bz2
conda install pytorch3d-0.7.8-py39_cu118_pyt241.tar.bz2
pip install opencv-python
```

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