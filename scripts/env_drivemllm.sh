
cd path/to/Drive-MLLM-main
pip install -r requirements.txt

# setup PYTHONPATH
echo 'export PYTHONPATH=$(pwd):$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc

# qwen-VL
cd /path/to/Qwen2.5-VL-main/
pip install -r requirements_web_demo.txt
pip install qwen-vl-utils[decord]
cd ../..

# setup PYTHONPATH
echo 'export PYTHONPATH=$(pwd):$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc


cd path/to/Drive-MLLM-main
# bash third_party/env_yoloworld.sh
## flash atten (optional)
pip install flash-attn --no-build-isolation --no-cache-dir
# download whl and install

