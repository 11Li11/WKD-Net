# WKD-Net
# Environments
conda create -n your_env_name python=3.10.13

conda activate your_env_name

pip3 install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

pip3 install timm==0.9.16 tensorboardX einops torchprofile fvcore==0.1.5.post20221221 triton==2.1.0

pip install causal_conv1d==1.1.3

pip install mamba_ssm==1.1.1
