## dependency
```bash
conda create -n FovLight python=3.9
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
wget https://anaconda.org/pytorch3d/pytorch3d/0.7.8/download/linux-64/pytorch3d-0.7.8-py39_cu118_pyt241.tar.bz2
conda install pytorch3d-0.7.8-py39_cu118_pyt241.tar.bz2
pip install opencv-python accelerate tensorboard pyiqa lpips iopath mitsuba openpyxl colour-science pyfvvdp flip-evaluator
```

## train
```bash
# create new session
tmux new -s FovLight
./train.sh
# exist Ctrl + b -> d
# connect to session
tmux attach -t FovLight
# kill session
tmux kill-session -t FovLight
# show tensorboard
tensorboard --logdir=logs
```