from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor import oneshot

recipe = [
    #GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
]
# Apply quantization using the built in open_platypus dataset.
oneshot(
    #model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    model="./dist/models/Llama-3.1-70B-Instruct",
    #model="./dist/models/CodeLlama-34b-hf",
    #model="./dist/models/Llama-3.1-8B-Instruct",
    dataset="open_platypus",
    recipe=recipe,
    output_dir="./dist/models/Llama-3.1-70B-Instruct-W8A8",
    #output_dir="./dist/models/Llama-3.1-70B-Instruct-W4A16_ASYM",
    #output_dir="./dist/models/CodeLlama-34b-hf-W4A16_ASYM",
    #output_dir="./dist/models/Llama-3.1-8B-Instruct-W4A16_ASYM",
    #output_dir="./dist/models/Llama-3.1-8B-Instruct-W8A8",
    max_seq_length=2048,
    num_calibration_samples=512,
)

