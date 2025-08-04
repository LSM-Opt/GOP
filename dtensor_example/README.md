# 환경 설정
* Python 3.12.11
* `pip install -r requirements.txt`

# 실행 방법
## 모델 설정
* 코드에서 모델 폴더가 dtensor_example/ 하위에 있음을 가정함
  * `git clone https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct /dist/Llama-3.1-8B-Instruct`
  * `ln -s /dist/Llama-3.1-8B-Instruct ./`
## 환경 별 실행 스크립트
### Single Node, Singe GPU
* `TORCH_COMPILE_DEBUG=1 WORLD_SIZE=1 RANK=0 LOCAL_WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=0 MASTER_PORT=0 python llm_test.py`
* init_device_mesh 크기를 1로 수정 필요
### Single Node, Multi GPU
* `NCCL_DEBUG_SUBSYS=ALL NCCL_DEBUG=INFO python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=gpu llm_test.py`
* TorchInductor에서는 오류로 torch.compile 불가. 지원하는 backend가 따로 존재하는 것으로 보임(torch/xla, TorchTitan, Torch-TensorRT, ...)
### Multi Nodes, Singe GPU
* Master: `TORCH_COMPILE_DEBUG=1 NCCL_DEBUG_SUBSYS=ALL NCCL_DEBUG=INFO NCCL_SOCKET_IFNAME=tun0 WORLD_SIZE=2 RANK=0 LOCAL_WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=12356 python llm_test.py`
* Workers: `TORCH_COMPILE_DEBUG=1 NCCL_DEBUG_SUBSYS=ALL NCCL_DEBUG=INFO NCCL_SOCKET_IFNAME=tun0 WORLD_SIZE=2 RANK=1 LOCAL_WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=[target] MASTER_PORT=12356 python llm_test.py`
## 디버깅 환경변수
* 참고: [https://docs.pytorch.org/docs/stable/distributed.html#debugging-torch-distributed-applications](https://docs.pytorch.org/docs/stable/distributed.html#debugging-torch-distributed-applications)
* `NCCL_DEBUG_SUBSYS=ALL NCCL_DEBUG=INFO`
  * NCCL 관련 실행정보 확인. Collective 연산 실행 등 런타임에 확인 가능.
* `TORCH_COMPILE_DEBUG`
  * torch.compile 사용 시, 컴파일 로그 저장.
  * TorchInductor backend에서는 torch_compile_debug/\[target\]torchinductor/fx_graph_readable.py 등 파일 위주로 확인