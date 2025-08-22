import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.tensor import distribute_tensor
from torch.profiler import profile, ProfilerActivity, schedule

### 디바이스 설정
torch.manual_seed(1000)

world_size = int(os.environ["WORLD_SIZE"])
rank = int(os.environ["RANK"])
local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
torch.cuda.set_device(rank)

### 모델 생성
MODEL="llama"
def init_model(name):
    if name == "resnet":
        import torchvision.models
        inp = generate_data(batch)[0]
        return torchvision.models.resnet50().to(torch.float32).cuda(), inp, None
    elif name == "llama":
        from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
        #base_model = './dist/models/Llama-3.1-8B-Instruct'
        #base_model = './dist/models/Llama-3.1-8B-Instruct-W4A16_ASYM'
        #base_model = './dist/models/Llama-3.1-8B-Instruct-NVFP4A16'
        base_model = './dist/models/CodeLlama-34b-hf-W4A16_ASYM'
        #base_model = './dist/models/Llama-3.1-70B-Instruct-W4A16_ASYM'
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype="auto",
        )
        
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        inp = tokenizer("hello, how are you today?", return_tensors="pt")
        
        from compressed_tensors.linear.compressed_linear import CompressedLinear
        for name, module in model.named_modules():
            if isinstance(module, CompressedLinear):
                print(name, module.quantization_status)
                
                from torch import Tensor
                from torch.nn import Parameter
                from torch.nn.functional import linear
                from compressed_tensors.quantization import QuantizationStatus
                def persist_forward(self, input: Tensor) -> Tensor:
                    if self.quantization_status != QuantizationStatus.COMPRESSED:
                        raise ValueError("Quantization status is not compressed")
                    
                    weight_data = self.compressor.decompress_module(self)
                    param = Parameter(weight_data, requires_grad=False)

                    return linear(input, param, self.bias)
                persist_forward = persist_forward.__get__(module, module.__class__)
                setattr(module, "forward", persist_forward)
                from compressed_tensors.utils import register_offload_parameter
                register_offload_parameter(module, "weight", module.weight_packed)

        output_example = None #model(input_ids=inp['input_ids'], attention_mask=inp['attention_mask'])
        return model, inp, tokenizer, output_example

model, inp, tokenizer, output_example = init_model(MODEL)
print("Original model")
print(model)
evice = "cuda"
model = model.eval()

### Distributed 설정
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel, SequenceParallel
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard

dist.init_process_group(backend="nccl")
torch.cuda.synchronize()

tp_mesh = init_device_mesh("cuda", (world_size,))

case_apply_parallel = True
case_attn = 0
case_mlp = 0
with torch.no_grad():
    for name, module in model.named_modules():
        print(f"Name: {name}")
        if module._get_name() == "LlamaDecoderLayer":
            def attention_parallelization(module, case_attn):
                if case_attn == 0:
                    if "2.7" in torch.__version__:
                        module.self_attn.num_heads = module.self_attn.num_heads // tp_mesh.size()
                        module.self_attn.num_key_value_heads = module.self_attn.num_key_value_heads // tp_mesh.size()
                    module_self_attn_parallel = parallelize_module(module.self_attn, tp_mesh,
                                                            {"q_proj": ColwiseParallel(),
                                                                "k_proj": ColwiseParallel(),
                                                                "v_proj": ColwiseParallel(),
                                                                "o_proj": RowwiseParallel()})
                    module_self_attn_parallel.q_proj.weight_shape = nn.Parameter(module_self_attn_parallel.q_proj.weight_shape.full_tensor().to("cpu"), requires_grad=False) # NotImplementedError: Operator aten.unbind.int does not have a sharding strategy registered.
                    module_self_attn_parallel.k_proj.weight_shape = nn.Parameter(module_self_attn_parallel.k_proj.weight_shape.full_tensor().to("cpu"), requires_grad=False)
                    module_self_attn_parallel.v_proj.weight_shape = nn.Parameter(module_self_attn_parallel.v_proj.weight_shape.full_tensor().to("cpu"), requires_grad=False)
                    module_self_attn_parallel.o_proj.weight_shape = nn.Parameter(module_self_attn_parallel.o_proj.weight_shape.full_tensor().to("cpu"), requires_grad=False)
                    delattr(module_self_attn_parallel.q_proj, "weight") # weight는 parallelize_module에서 weight_packed sharding 유도를 위해서만 사용. 이후 사용되지 않아 제거.
                    delattr(module_self_attn_parallel.k_proj, "weight")
                    delattr(module_self_attn_parallel.v_proj, "weight")
                    delattr(module_self_attn_parallel.o_proj, "weight")
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
                    module_mlp_parallel.gate_proj.weight_shape = nn.Parameter(module_mlp_parallel.gate_proj.weight_shape.full_tensor().to("cpu"), requires_grad=False)
                    module_mlp_parallel.up_proj.weight_shape = nn.Parameter(module_mlp_parallel.up_proj.weight_shape.full_tensor().to("cpu"), requires_grad=False)
                    module_mlp_parallel.down_proj.weight_shape = nn.Parameter(module_mlp_parallel.down_proj.weight_shape.full_tensor().to("cpu"), requires_grad=False)
                    delattr(module_mlp_parallel.gate_proj, "weight")
                    delattr(module_mlp_parallel.up_proj, "weight")
                    delattr(module_mlp_parallel.down_proj, "weight")
                elif case_mlp == 1: # GSPMD attempt 1 - slow
                    module_mlp_parallel = parallelize_module(module.mlp, tp_mesh, {"gate_proj": RowwiseParallel(input_layouts=Replicate()), "up_proj": RowwiseParallel(input_layouts=Replicate()), "down_proj": ColwiseParallel(output_layouts=Replicate())})
                else:
                    dist.destroy_process_group()
                    raise ValueError(f"Invalid mlp parallelization case: {case_mlp}")
                return module_mlp_parallel
            
            if case_attn != -1:
                module_self_attn_parallel = attention_parallelization(module, case_attn)
                module.self_attn = module_self_attn_parallel
            if case_mlp != -1:
                module_mlp_parallel = mlp_parallelization(module, case_mlp)
                module.mlp = module_mlp_parallel

    if case_apply_parallel:
        model.model = parallelize_module(model.model, tp_mesh,
                                {"embed_tokens": RowwiseParallel(input_layouts=Replicate()) })
    print("Parallelized model")
    print(model)

### 모델 컴파일
device = "cuda"
#model = model.eval().to(device)
model = model.to(device)

# 메모리 사용량 상세 분석
def analyze_memory_usage(model):
    # 모델 파라미터 크기
    param_size = 0
    size_dict = {}
    for name, parameter in model.named_parameters():
        size = (parameter.numel() * parameter.element_size())
        param_size += size
        size_dict[name] = size
    # order by size and print size
    sorted_size_dict = sorted(size_dict.items(), key=lambda x: x[1], reverse=True)
    for name, size in sorted_size_dict:
        print(name, ",", size)
    print(f"모델 파라미터: {param_size / 1024**3:.2f} GB")
    
    # GPU 메모리 현황
    print(f"할당된 메모리: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"캐시된 메모리: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    # 메모리 상세 정보
    print(torch.cuda.memory_summary())

analyze_memory_usage(model)

import torch._dynamo
torch._dynamo.reset()

use_compile = False
if use_compile:
    #import torch_tensorrt
    #model_opt = torch.compile(model, backend="tensorrt") #model
    #model_opt = torch.compile(model, mode="max-autotune-no-cudagraphs", dynamic=False)     # can't create trace file
    #model_opt = torch.compile(model, dynamic=False)     # can't create trace file
    #model_opt = torch.compile(model, mode="reduce-overhead", dynamic=False)                # cudagraph error
    model_opt = torch.compile(model, mode="reduce-overhead")                                # Fake tensor error or no trace file
    print("Use compiled model")
else:
    model_opt = model
    print("Use original model")
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

#analyze_parallel_structure(model)

### 모델 실행
trace_file = f"trace_{rank}.json"
with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
            on_trace_ready=lambda prof: prof.export_chrome_trace(trace_file),
            schedule=torch.profiler.schedule(skip_first=2, wait=1, warmup=1, active=1, repeat=1)
        ) as prof:
    input_ids = inp['input_ids'].to(device)
    attention_mask = inp['attention_mask'].to(device)

    generate_length = 20 #500
    with torch.no_grad():
        phase = "prefill" # "prefill" or "decode"
                    
        from tqdm import tqdm
        if phase == "prefill":
            for i in tqdm(range(generate_length), desc="Prefill iterations"):
                dist.broadcast(input_ids, src=0)
                dist.broadcast(attention_mask, src=0)
                torch.cuda.synchronize()
                dist.barrier()

                output = model_opt(input_ids=input_ids, attention_mask=attention_mask)
                if rank == 0:
                    output_ids = output.logits.argmax(dim=-1)
                    print(tokenizer.decode(output_ids[-1, -1:], skip_special_tokens=True))
                prof.step()
        elif phase == "decode":
            if case_attn != 0 and case_mlp != 0:
                # Due to splitting of key-value cache across GPUs via head axis (Colwise->Rowwise)
                # TODO: support this case
                dist.destroy_process_group()
                raise ValueError("case_attn and case_mlp cannot be both 0 when decode phase is evaluated")
            
            past_key_values = tuple(tuple(torch.chunk(tensor, world_size, dim=1)[rank % world_size].to(device) for tensor in layer) for layer in output_example.past_key_values)
            output_ids = output_example.logits.argmax(dim=-1).to(device)
            input_ids = torch.concat([input_ids, output_ids[:, -1:]], dim=-1)
            current_input_ids = input_ids[:, -1:]
            attention_mask = torch.ones_like(input_ids)
            for i in tqdm(range(generate_length), desc="Decoding iterations"):
                dist.broadcast(current_input_ids, src=0)
                dist.broadcast(attention_mask, src=0)
                torch.cuda.synchronize()
                dist.barrier()

                output = model_opt(input_ids=current_input_ids,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True)
                if rank == 0:
                    output_ids = output.logits.argmax(dim=-1)
                    print(tokenizer.decode(output_ids[-1, -1:], skip_special_tokens=True))
                prof.step()

if rank == 0:
    predicted_ids = torch.argmax(output.logits, dim=-1)
    decoded_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
    print(decoded_text)

print(prof.events())

dist.destroy_process_group()
