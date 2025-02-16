from datasets import load_dataset
import torch
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from rewards import format_reward, accuracy_reward

SYSTEM_PROMPT = """
A conversation between User and Assistant. The User asks a question, and the Assistant solves it.
The Assistant first reasons internally before responding.

Respond in the following format:
<reasoning>
Your step-by-step thought process here.
</reasoning>
<answer>
Your final answer here.
</answer>
"""

train_dataset, test_dataset = load_dataset("AI-MO/NuminaMath-TIR", split=["train[:5%]", "test[:5%]"])

def make_conversation(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
    }

train_dataset = train_dataset.map(make_conversation)
test_dataset = test_dataset.map(make_conversation)
train_dataset = train_dataset.remove_columns(["messages", "problem"])

# Baseline model
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
output_dir = "math_solver_model"
run_name="Qwen-0.5B-GRPO"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=None
).to("cuda")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Training arguments + training
training_args = GRPOConfig(
    output_dir=output_dir,
    run_name=run_name,
    learning_rate=5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type='cosine',
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=16,
    max_prompt_length=256,
    max_completion_length=200,
    num_train_epochs=1,
    save_steps=100,
    max_grad_norm=0.1,
    log_on_each_node=False,
    use_vllm=True,
    vllm_gpu_memory_utilization=.3,
    vllm_device="cuda:0",
    report_to="none"
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        format_reward,
        accuracy_reward],
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()