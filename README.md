### SafeRLHF复现日志

#### 一些问题

```bash
conda create -n py311 python=3.11
conda activate py311
python -m pip install -r requirements.txt
export WANDB_API_KEY="$MY_KEY"#wandb官网获取
```

##### 1.Tokenizer

```
line 822, in from_pretrained
    raise ValueError(
ValueError: Tokenizer class LLaMATokenizer does not exist or is not currently imported.
```

修改 `../llama-7b-hf/tokenizer_config.json`文件为：

```
{
    "bos_token": "<s>",
    "eos_token": "</s>",
    "model_max_length": 1000000000000000019884624838656,
    "tokenizer_class": "LlamaTokenizer",
    "unk_token": "<unk>"
}
```

##### 2.alpaca数据集无法下载的问题

```
File "/home/bzx_yjy/code/safe-rlhf/safe_rlhf/datasets/base.py", line 163, in load
    return cls(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^
  File "/home/bzx_yjy/code/safe-rlhf/safe_rlhf/datasets/raw/alpaca.py", line 31, in __init__
    self.data = load_dataset(path or 'tatsu-lab/alpaca', split='train')
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/bzx_yjy/miniconda3/envs/py311/lib/python3.11/site-packages/datasets/load.py", line 2548, in load_dataset
    builder_instance = load_dataset_builder(
                       ^^^^^^^^^^^^^^^^^^^^^
  File "/home/bzx_yjy/miniconda3/envs/py311/lib/python3.11/site-packages/datasets/load.py", line 2220, in load_dataset_builder
    dataset_module = dataset_module_factory(
                     ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/bzx_yjy/miniconda3/envs/py311/lib/python3.11/site-packages/datasets/load.py", line 1871, in dataset_module_factory
    raise e1 from None
  File "/home/bzx_yjy/miniconda3/envs/py311/lib/python3.11/site-packages/datasets/load.py", line 1805, in dataset_module_factory
    raise ConnectionError(f"Couldn't reach '{path}' on the Hub ({type(e).__name__})")
ConnectionError: Couldn't reach 'tatsu-lab/alpaca' on the Hub (ConnectionError)
```

解决方法：

[这可能是全网最好解决中国hugggingface.co无法访问问题 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/627688602)

```python
export HF_ENDPOINT=https://hf-mirror.com
bash scripts/sft.sh --model_name_or_path ../llama-7b-hf --output_dir output/sft
```

也可以将数据集下载到本地加载(修改`alpaca.py`)

```
vim /home/bzx_yjy/code/safe-rlhf/safe_rlhf/datasets/raw/alpaca.py
```

将alpaca.py中的load dataset的路径改为 `self.data = load_dataset('/home/bzx_yjy/code/alpaca')['train']`

##### 3.gcc版本过高编译失败

[替换系统 gcc 为 anaconda 安装的 gcc - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/662208106)

```
conda install -c conda-forge gcc_linux-64==8.5.0
conda install -c conda-forge gxx_impl_linux-64==8.5.0
```



##### 4.git-lfs下载（无管理员权限）+ llama模型下载

```bash
tar -zxvf  git-lfs-linux-amd64-v3.4.1.tar.gz
vi git-lfs-3.4.1/install.sh #修改下路径
. git-lfs-3.4.1/install.sh
```

下载llama模型：

```
git clone https://gitee.com/modelee/llama-7b-hf.git
```



有权限的话直接装：

```
git lfs install
```



##### 5.AttributeError: 'DeepSpeedCPUAdam' object has no attribute 'ds_opt_adam'

[Deepspeed多卡多机ZeRo-3训练踩的坑 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/642819809)

```
vim /home/bzx_yjy/code/safe-rlhf/safe_rlhf/trainers/supervised_trainer.py
```



##### 6.

这是什么操作……

[Installed CUDA version 12.1 does not match the version torch was compiled with 11.7_attributeerror: 'deepspeedcpuadam' object has no a-CSDN博客](https://blog.csdn.net/gs80140/article/details/130848792)



##### 7.



[[BUG\]exits with return code = -9 · Issue #4181 · microsoft/DeepSpeed (github.com)](https://github.com/microsoft/DeepSpeed/issues/4181)

这个错误会发生在开启deepspeed时（`--offload` all/parameter/optimizer)，原因应该是OOM，不开的话显存又会爆。



最后解决方法是：在2张3090的平台上不管怎么调batch_size和--offload的参数都没法训练，所以在4张3090的平台上训练(SFT)。

```
$bash scripts/sft.sh --model_name_or_path ../llama-7b-hf --output_dir output/sft --offload all

log:

+++ dirname scripts/sft.sh
++ cd scripts
++ pwd
+ SCRIPT_DIR=/home/webace/Tmp/test0229/safe-rlhf/scripts
++ dirname /home/webace/Tmp/test0229/safe-rlhf/scripts
+ ROOT_DIR=/home/webace/Tmp/test0229/safe-rlhf
+ export PYTHONPATH=/home/webace/Tmp/test0229/safe-rlhf
+ PYTHONPATH=/home/webace/Tmp/test0229/safe-rlhf
+ export LOGLEVEL=WARNING
+ LOGLEVEL=WARNING
+ MODEL_NAME_OR_PATH=huggyllama/llama-7b
+ OUTPUT_DIR=/home/webace/Tmp/test0229/safe-rlhf/output/sft
+ unset HOSTFILE
+ ZERO_STAGE=3
+ OFFLOAD=none
+ [[ 6 -gt 0 ]]
+ arg=--model_name_or_path
+ shift
+ case "${arg}" in
+ MODEL_NAME_OR_PATH=../llama-7b-hf
+ shift
+ [[ 4 -gt 0 ]]
+ arg=--output_dir
+ shift
+ case "${arg}" in
+ OUTPUT_DIR=output/sft
+ shift
+ [[ 2 -gt 0 ]]
+ arg=--offload
+ shift
+ case "${arg}" in
+ OFFLOAD=all
+ shift
+ [[ 0 -gt 0 ]]
+ mkdir -p output/sft
++ cd output/sft
++ pwd
+ OUTPUT_DIR=/home/webace/Tmp/test0229/safe-rlhf/output/sft
+ [[ ! -f /home/webace/Tmp/test0229/safe-rlhf/output/sft/.gitignore ]]
+ cp -f scripts/sft.sh /home/webace/Tmp/test0229/safe-rlhf/output/sft/script.sh
+ [[ -z '' ]]
+ export WANDB_MODE=offline
+ WANDB_MODE=offline
+ MASTER_PORT_START=10000
+ MASTER_PORT_END=65535
++ shuf
++ head -n 1
++ comm -23 /dev/fd/63 /dev/fd/62
+++ seq 10000 65535
+++ sort
+++ ss -Htan
+++ awk '{ print $4 }'
+++ awk -F : '{ print $NF }'
+++ sort -u
+ MASTER_PORT=29451
+ DEEPSPEED_ARGS=()
+ [[ -n '' ]]
+ DEEPSPEED_ARGS+=("--master_port" "${MASTER_PORT}")
+ exec
++ tee /home/webace/Tmp/test0229/safe-rlhf/output/sft/stdout.log
++ tee /home/webace/Tmp/test0229/safe-rlhf/output/sft/stderr.log
+ deepspeed --master_port 29451 --module safe_rlhf.finetune --train_datasets alpaca --model_name_or_path ../llama-7b-hf --max_length 512 --trust_remote_code True --epochs 3 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 8 --gradient_checkpointing --learning_rate 2e-5 --lr_scheduler_type cosine --lr_warmup_ratio 0.03 --weight_decay 0.0 --seed 42 --output_dir /home/webace/Tmp/test0229/safe-rlhf/output/sft --log_type wandb --log_project Safe-RLHF-SFT --zero_stage 3 --offload all --bf16 True --tf32 True
[2024-02-29 19:29:25,377] [INFO] [real_accelerator.py:191:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2024-02-29 19:29:25,864] [WARNING] [runner.py:202:fetch_hostfile] Unable to find hostfile, will proceed with training with local resources only.
[2024-02-29 19:29:25,864] [INFO] [runner.py:568:main] cmd = /data/Development/anaconda3/envs/py311/bin/python -u -m deepspeed.launcher.launch --world_info=eyJsb2NhbGhvc3QiOiBbMCwgMSwgMiwgM119 --master_addr=127.0.0.1 --master_port=29451 --module --enable_each_rank_log=None safe_rlhf.finetune --train_datasets alpaca --model_name_or_path ../llama-7b-hf --max_length 512 --trust_remote_code True --epochs 3 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 8 --gradient_checkpointing --learning_rate 2e-5 --lr_scheduler_type cosine --lr_warmup_ratio 0.03 --weight_decay 0.0 --seed 42 --output_dir /home/webace/Tmp/test0229/safe-rlhf/output/sft --log_type wandb --log_project Safe-RLHF-SFT --zero_stage 3 --offload all --bf16 True --tf32 True
[2024-02-29 19:29:27,811] [INFO] [real_accelerator.py:191:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2024-02-29 19:29:28,328] [INFO] [launch.py:145:main] WORLD INFO DICT: {'localhost': [0, 1, 2, 3]}
[2024-02-29 19:29:28,328] [INFO] [launch.py:151:main] nnodes=1, num_local_procs=4, node_rank=0
[2024-02-29 19:29:28,328] [INFO] [launch.py:162:main] global_rank_mapping=defaultdict(<class 'list'>, {'localhost': [0, 1, 2, 3]})
[2024-02-29 19:29:28,328] [INFO] [launch.py:163:main] dist_world_size=4
[2024-02-29 19:29:28,328] [INFO] [launch.py:165:main] Setting CUDA_VISIBLE_DEVICES=0,1,2,3
[2024-02-29 19:29:28,329] [INFO] [launch.py:253:main] process 1149977 spawned with command: ['/data/Development/anaconda3/envs/py311/bin/python', '-u', '-m', 'safe_rlhf.finetune', '--local_rank=0', '--train_datasets', 'alpaca', '--model_name_or_path', '../llama-7b-hf', '--max_length', '512', '--trust_remote_code', 'True', '--epochs', '3', '--per_device_train_batch_size', '1', '--per_device_eval_batch_size', '1', '--gradient_accumulation_steps', '8', '--gradient_checkpointing', '--learning_rate', '2e-5', '--lr_scheduler_type', 'cosine', '--lr_warmup_ratio', '0.03', '--weight_decay', '0.0', '--seed', '42', '--output_dir', '/home/webace/Tmp/test0229/safe-rlhf/output/sft', '--log_type', 'wandb', '--log_project', 'Safe-RLHF-SFT', '--zero_stage', '3', '--offload', 'all', '--bf16', 'True', '--tf32', 'True']
[2024-02-29 19:29:28,329] [INFO] [launch.py:253:main] process 1149978 spawned with command: ['/data/Development/anaconda3/envs/py311/bin/python', '-u', '-m', 'safe_rlhf.finetune', '--local_rank=1', '--train_datasets', 'alpaca', '--model_name_or_path', '../llama-7b-hf', '--max_length', '512', '--trust_remote_code', 'True', '--epochs', '3', '--per_device_train_batch_size', '1', '--per_device_eval_batch_size', '1', '--gradient_accumulation_steps', '8', '--gradient_checkpointing', '--learning_rate', '2e-5', '--lr_scheduler_type', 'cosine', '--lr_warmup_ratio', '0.03', '--weight_decay', '0.0', '--seed', '42', '--output_dir', '/home/webace/Tmp/test0229/safe-rlhf/output/sft', '--log_type', 'wandb', '--log_project', 'Safe-RLHF-SFT', '--zero_stage', '3', '--offload', 'all', '--bf16', 'True', '--tf32', 'True']
[2024-02-29 19:29:28,330] [INFO] [launch.py:253:main] process 1149979 spawned with command: ['/data/Development/anaconda3/envs/py311/bin/python', '-u', '-m', 'safe_rlhf.finetune', '--local_rank=2', '--train_datasets', 'alpaca', '--model_name_or_path', '../llama-7b-hf', '--max_length', '512', '--trust_remote_code', 'True', '--epochs', '3', '--per_device_train_batch_size', '1', '--per_device_eval_batch_size', '1', '--gradient_accumulation_steps', '8', '--gradient_checkpointing', '--learning_rate', '2e-5', '--lr_scheduler_type', 'cosine', '--lr_warmup_ratio', '0.03', '--weight_decay', '0.0', '--seed', '42', '--output_dir', '/home/webace/Tmp/test0229/safe-rlhf/output/sft', '--log_type', 'wandb', '--log_project', 'Safe-RLHF-SFT', '--zero_stage', '3', '--offload', 'all', '--bf16', 'True', '--tf32', 'True']
[2024-02-29 19:29:28,331] [INFO] [launch.py:253:main] process 1149980 spawned with command: ['/data/Development/anaconda3/envs/py311/bin/python', '-u', '-m', 'safe_rlhf.finetune', '--local_rank=3', '--train_datasets', 'alpaca', '--model_name_or_path', '../llama-7b-hf', '--max_length', '512', '--trust_remote_code', 'True', '--epochs', '3', '--per_device_train_batch_size', '1', '--per_device_eval_batch_size', '1', '--gradient_accumulation_steps', '8', '--gradient_checkpointing', '--learning_rate', '2e-5', '--lr_scheduler_type', 'cosine', '--lr_warmup_ratio', '0.03', '--weight_decay', '0.0', '--seed', '42', '--output_dir', '/home/webace/Tmp/test0229/safe-rlhf/output/sft', '--log_type', 'wandb', '--log_project', 'Safe-RLHF-SFT', '--zero_stage', '3', '--offload', 'all', '--bf16', 'True', '--tf32', 'True']
[2024-02-29 19:29:30,326] [INFO] [real_accelerator.py:191:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2024-02-29 19:29:30,328] [INFO] [real_accelerator.py:191:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2024-02-29 19:29:30,333] [INFO] [real_accelerator.py:191:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2024-02-29 19:29:30,414] [INFO] [real_accelerator.py:191:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2024-02-29 19:29:31,620] [INFO] [comm.py:637:init_distributed] cdb=None
[2024-02-29 19:29:31,620] [INFO] [comm.py:668:init_distributed] Initializing TorchBackend in DeepSpeed with backend nccl
[2024-02-29 19:29:31,657] [INFO] [comm.py:637:init_distributed] cdb=None
[2024-02-29 19:29:31,673] [INFO] [comm.py:637:init_distributed] cdb=None
[2024-02-29 19:29:31,895] [INFO] [comm.py:637:init_distributed] cdb=None
Set logger level to WARNING.
Loading checkpoint shards: 100%|██████████| 33/33 [00:17<00:00,  1.88it/s]
Loading checkpoint shards: 100%|██████████| 33/33 [00:17<00:00,  1.88it/s]
Loading checkpoint shards: 100%|██████████| 33/33 [00:17<00:00,  1.88it/s]
Loading checkpoint shards: 100%|██████████| 33/33 [00:17<00:00,  1.88it/s]
You are using the default legacy behaviour of the <class 'transformers.models.llama.tokenization_llama.LlamaTokenizer'>. This is expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you. If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it means, and thoroughly read the reason why this was added as explained in https://github.com/huggingface/transformers/pull/24565
You are using the default legacy behaviour of the <class 'transformers.models.llama.tokenization_llama.LlamaTokenizer'>. This is expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you. If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it means, and thoroughly read the reason why this was added as explained in https://github.com/huggingface/transformers/pull/24565
You are using the default legacy behaviour of the <class 'transformers.models.llama.tokenization_llama.LlamaTokenizer'>. This is expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you. If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it means, and thoroughly read the reason why this was added as explained in https://github.com/huggingface/transformers/pull/24565
You are using the default legacy behaviour of the <class 'transformers.models.llama.tokenization_llama.LlamaTokenizer'>. This is expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you. If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it means, and thoroughly read the reason why this was added as explained in https://github.com/huggingface/transformers/pull/24565
Installed CUDA version 12.2 does not match the version torch was compiled with 12.1 but since the APIs are compatible, accepting this combination
Using /home/webace/.cache/torch_extensions/py311_cu121 as PyTorch extensions root...
Creating extension directory /home/webace/.cache/torch_extensions/py311_cu121/cpu_adam...
Detected CUDA files, patching ldflags
Emitting ninja build file /home/webace/.cache/torch_extensions/py311_cu121/cpu_adam/build.ninja...
Building extension module cpu_adam...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
Installed CUDA version 12.2 does not match the version torch was compiled with 12.1 but since the APIs are compatible, accepting this combination
Using /home/webace/.cache/torch_extensions/py311_cu121 as PyTorch extensions root...
Installed CUDA version 12.2 does not match the version torch was compiled with 12.1 but since the APIs are compatible, accepting this combination
Using /home/webace/.cache/torch_extensions/py311_cu121 as PyTorch extensions root...
Installed CUDA version 12.2 does not match the version torch was compiled with 12.1 but since the APIs are compatible, accepting this combination
Using /home/webace/.cache/torch_extensions/py311_cu121 as PyTorch extensions root...
[1/4] /usr/local/cuda/bin/nvcc --generate-dependencies-with-compile --dependency-output custom_cuda_kernel.cuda.o.d -DTORCH_EXTENSION_NAME=cpu_adam -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" -I/data/Development/anaconda3/envs/py311/lib/python3.11/site-packages/deepspeed/ops/csrc/includes -I/usr/local/cuda/include -isystem /data/Development/anaconda3/envs/py311/lib/python3.11/site-packages/torch/include -isystem /data/Development/anaconda3/envs/py311/lib/python3.11/site-packages/torch/include/torch/csrc/api/include -isystem /data/Development/anaconda3/envs/py311/lib/python3.11/site-packages/torch/include/TH -isystem /data/Development/anaconda3/envs/py311/lib/python3.11/site-packages/torch/include/THC -isystem /usr/local/cuda/include -isystem /data/Development/anaconda3/envs/py311/include/python3.11 -D_GLIBCXX_USE_CXX11_ABI=0 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -gencode=arch=compute_86,code=compute_86 -gencode=arch=compute_86,code=sm_86 --compiler-options '-fPIC' -O3 --use_fast_math -std=c++17 -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ -U__CUDA_NO_HALF2_OPERATORS__ --threads=8 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_86,code=compute_86 -DBF16_AVAILABLE -U__CUDA_NO_BFLOAT16_OPERATORS__ -U__CUDA_NO_BFLOAT162_OPERATORS__ -c /data/Development/anaconda3/envs/py311/lib/python3.11/site-packages/deepspeed/ops/csrc/common/custom_cuda_kernel.cu -o custom_cuda_kernel.cuda.o
[2/4] c++ -MMD -MF cpu_adam_impl.o.d -DTORCH_EXTENSION_NAME=cpu_adam -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" -I/data/Development/anaconda3/envs/py311/lib/python3.11/site-packages/deepspeed/ops/csrc/includes -I/usr/local/cuda/include -isystem /data/Development/anaconda3/envs/py311/lib/python3.11/site-packages/torch/include -isystem /data/Development/anaconda3/envs/py311/lib/python3.11/site-packages/torch/include/torch/csrc/api/include -isystem /data/Development/anaconda3/envs/py311/lib/python3.11/site-packages/torch/include/TH -isystem /data/Development/anaconda3/envs/py311/lib/python3.11/site-packages/torch/include/THC -isystem /usr/local/cuda/include -isystem /data/Development/anaconda3/envs/py311/include/python3.11 -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC -std=c++17 -O3 -std=c++17 -g -Wno-reorder -L/usr/local/cuda/lib64 -lcudart -lcublas -g -march=native -fopenmp -D__AVX512__ -D__ENABLE_CUDA__ -DBF16_AVAILABLE -c /data/Development/anaconda3/envs/py311/lib/python3.11/site-packages/deepspeed/ops/csrc/adam/cpu_adam_impl.cpp -o cpu_adam_impl.o
[3/4] c++ -MMD -MF cpu_adam.o.d -DTORCH_EXTENSION_NAME=cpu_adam -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" -I/data/Development/anaconda3/envs/py311/lib/python3.11/site-packages/deepspeed/ops/csrc/includes -I/usr/local/cuda/include -isystem /data/Development/anaconda3/envs/py311/lib/python3.11/site-packages/torch/include -isystem /data/Development/anaconda3/envs/py311/lib/python3.11/site-packages/torch/include/torch/csrc/api/include -isystem /data/Development/anaconda3/envs/py311/lib/python3.11/site-packages/torch/include/TH -isystem /data/Development/anaconda3/envs/py311/lib/python3.11/site-packages/torch/include/THC -isystem /usr/local/cuda/include -isystem /data/Development/anaconda3/envs/py311/include/python3.11 -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC -std=c++17 -O3 -std=c++17 -g -Wno-reorder -L/usr/local/cuda/lib64 -lcudart -lcublas -g -march=native -fopenmp -D__AVX512__ -D__ENABLE_CUDA__ -DBF16_AVAILABLE -c /data/Development/anaconda3/envs/py311/lib/python3.11/site-packages/deepspeed/ops/csrc/adam/cpu_adam.cpp -o cpu_adam.o
[4/4] c++ cpu_adam.o cpu_adam_impl.o custom_cuda_kernel.cuda.o -shared -lcurand -L/data/Development/anaconda3/envs/py311/lib/python3.11/site-packages/torch/lib -lc10 -lc10_cuda -ltorch_cpu -ltorch_cuda -ltorch -ltorch_python -L/usr/local/cuda/lib64 -lcudart -o cpu_adam.so
Loading extension module cpu_adam...
Time to load cpu_adam op: 31.080963611602783 seconds
Loading extension module cpu_adam...
Loading extension module cpu_adam...
Time to load cpu_adam op: 31.015425443649292 seconds
Time to load cpu_adam op: 31.019861459732056 seconds
Loading extension module cpu_adam...
Time to load cpu_adam op: 31.03505301475525 seconds
Parameter Offload: Total persistent parameters: 266240 in 65 params
`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
wandb: Tracking run with wandb version 0.16.3
wandb: W&B syncing is set to `offline` in this directory.
wandb: Run `wandb online` or set WANDB_MODE=online to enable cloud syncing.
***** Running training *****
Training 1/3 epoch:   0%|          | 0/39003 [00:00<?, ?it/s]`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
Training 1/3 epoch (loss 1.1828):   0%|          | 115/39003 [09:28<53:15:37,  4.93s/it]
```



##### 8.pip安装mpi4py失败

[Failed to build mpi4py ERROR: Could not build wheels for mpi4py, which is required to install pyproj-CSDN博客](https://blog.csdn.net/Clown_pan/article/details/128307006)

用conda装



#### to-do

1、跑通RLHF过程

2、仔细阅读代码，思考能不能改进or集成其它的安全强化学习算法进入代码中


