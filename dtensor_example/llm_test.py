import os
import torch
import torch.distributed as dist
from torch.profiler import profile, ProfilerActivity

### 디바이스 설정
torch.manual_seed(1000)

world_size = int(os.environ["WORLD_SIZE"])
rank = int(os.environ["RANK"])
local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

### 모델 생성
MODEL="llama"
def init_model(name):
    if name == "resnet":
        import torchvision.models
        inp = generate_data(batch)[0]
        return torchvision.models.resnet50().to(torch.float32).cuda(), inp, None
    elif name == "llama":
        from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
        base_model = './Llama-3.1-8B-Instruct'
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        inp = tokenizer("hello, how are you today?", return_tensors="pt")
        return model, inp, tokenizer

model, inp, tokenizer = init_model(MODEL)
print("Original model")
print(model)

### Distributed 설정
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel, SequenceParallel
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard

dist.init_process_group(backend="nccl")
torch.cuda.synchronize()

tp_mesh = init_device_mesh("cuda", (world_size,))

for name, module in model.named_modules():
    print(f"Name: {name}")
    if module._get_name() == "LlamaDecoderLayer":
        case_attn = 0
        case_mlp = 0
        def attention_parallelization(module, case_attn):
            if case_attn == 0:
                module.self_attn.num_heads = module.self_attn.num_heads // tp_mesh.size()
                module.self_attn.num_key_value_heads = module.self_attn.num_key_value_heads // tp_mesh.size()
                module_self_attn_parallel = parallelize_module(module.self_attn, tp_mesh,
                                                        {"q_proj": ColwiseParallel(),
                                                            "k_proj": ColwiseParallel(),
                                                            "v_proj": ColwiseParallel(),
                                                            "o_proj": RowwiseParallel()})
            elif case_attn == 1:
                module_self_attn_parallel = parallelize_module(module.self_attn, tp_mesh,
                                                        {"q_proj": RowwiseParallel(input_layouts=Replicate()),
                                                            "k_proj": RowwiseParallel(input_layouts=Replicate()),
                                                            "v_proj": RowwiseParallel(input_layouts=Replicate()),
                                                            "o_proj": RowwiseParallel(input_layouts=Replicate())})
            elif case_attn == 2:
                module.self_attn.num_heads = module.self_attn.num_heads // tp_mesh.size()
                module.self_attn.num_key_value_heads = module.self_attn.num_key_value_heads // tp_mesh.size()
                module_self_attn_parallel = parallelize_module(module.self_attn, tp_mesh,
                                                        {"q_proj": ColwiseParallel(),
                                                            "k_proj": ColwiseParallel(),
                                                            "v_proj": ColwiseParallel(),
                                                            "o_proj": ColwiseParallel(input_layouts=Shard(dim=-1), output_layouts=Replicate())})
            else:
                dist.destroy_process_group()
                raise ValueError(f"Invalid attention parallelization case: {case_attn}")
            return module_self_attn_parallel
        
        def mlp_parallelization(module, case_mlp):
            if case_mlp == 0: # GSPMD attempt 2, 3 - fast
                module_mlp_parallel = parallelize_module(module.mlp, tp_mesh, {"gate_proj": ColwiseParallel(), "up_proj": ColwiseParallel(), "down_proj": RowwiseParallel()})
            elif case_mlp == 1: # GSPMD attempt 1 - slow
                module_mlp_parallel = parallelize_module(module.mlp, tp_mesh, {"gate_proj": RowwiseParallel(input_layouts=Replicate()), "up_proj": RowwiseParallel(input_layouts=Replicate()), "down_proj": ColwiseParallel(output_layouts=Replicate())})
            else:
                dist.destroy_process_group()
                raise ValueError(f"Invalid mlp parallelization case: {case_mlp}")
            return module_mlp_parallel
        
        module_self_attn_parallel = attention_parallelization(module, case_attn)
        module_mlp_parallel = mlp_parallelization(module, case_mlp)
        module.mlp = module_mlp_parallel

model.model = parallelize_module(model.model, tp_mesh,
                           {"embed_tokens": RowwiseParallel(input_layouts=Replicate()) })
print("Parallelized model")
print(model)


### 모델 컴파일
device = "cuda"
model = model.to(device)

import torch._dynamo
torch._dynamo.reset()

# 
model_opt = torch.compile(model, backend="tensorrt") if local_world_size > 1 \
                else torch.compile(model, mode="reduce-overhead")
print("Compiled model")
print(model_opt)


### 모델 DTensor 정보 확인
def analyze_parallel_structure(model):
    print("Parallelized Model Structure Analysis:")

    for name, module in model.named_modules():
        if hasattr(module, '_ddp_params_and_buffers_to_ignore'):
            print(f"  {name}: DDP ignored parameters")

        if hasattr(module, 'device_mesh'):
            print(f"  {name}: Device mesh - {module.device_mesh}")

        if hasattr(module, 'placements'):
            print(f"  {name}: Placement strategy - {module.placements}")

        # DTensor 정보 확인
        for param_name, param in module.named_parameters(recurse=False):
            if hasattr(param, '_spec') and param._spec is not None:
                print(f"    {param_name}: DTensor spec - {param._spec}")
                print(f"      - Mesh: {param._spec.device_mesh}")
                print(f"      - Placements: {param._spec.placements}")

analyze_parallel_structure(model)

### 모델 실행
trace_file = "trace.json"
with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
            on_trace_ready=lambda prof: prof.export_chrome_trace(trace_file)
        ) as prof:
    input_ids = inp['input_ids'].to(device)
    attention_mask = inp['attention_mask'].to(device)
    dist.broadcast(input_ids, src=0)
    dist.broadcast(attention_mask, src=0)

    with torch.no_grad():
        # implement text generation without kv cache
        generate_length = 10
        for i in range(generate_length):
            output = model_opt(input_ids=input_ids, attention_mask=attention_mask)
            output_ids = output.logits.argmax(dim=-1)
            input_ids = torch.concat([input_ids, output_ids[:, -1:]], dim=-1)
            attention_mask = torch.ones_like(input_ids)
            dist.broadcast(input_ids, src=0)
            dist.broadcast(attention_mask, src=0)
            if rank == 0:
                print(tokenizer.decode(input_ids[-1], skip_special_tokens=True))

if rank == 0:
    predicted_ids = torch.argmax(output.logits, dim=-1)
    decoded_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
    print(decoded_text)

print(prof.events())

dist.destroy_process_group()
