import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
from unsloth import FastLanguageModel # uses backprop from output to input for training by using chain rule gradient descent. helps us compress from 32bit to 4 bit.
from datasets import load_dataset
from trl import SFTTrainer # lib for supervised fine tuning of language models with LoRA.
from transformers import TrainingArguments # arguments to configure the training parameters.
import torch 


max_seq_length = 1024 # max sequence length for the model, suitable for my 3060 GPU. Attention is O(n^2) so be mindful of this when training. 1024 is a sweet spot for now.
dtype = None # let unsloth decide the data type. bf16 or fp16 depending upon 3060.
load_in_4bit = True  # reducing newtowk weights to 4bits. (Heavy compression, minimal performance loss)


model, tokenizer = FastLanguageModel.from_pretrained( 
    model_name = "unsloth/Qwen2.5-Coder-3B-Instruct", # load the base model. We will fine tune it with LoRA on our special dataset.
    max_seq_length = max_seq_length, # limit the sequence length for training to save memory and speed up training. 1024 is a good balance for my GPU.
    dtype = dtype, 
    load_in_4bit = load_in_4bit, # 4bits quantization
)


model = FastLanguageModel.get_peft_model( 
    model,
    r = 16, # 16 as a rank is good starting point for peformance and efficiency.
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",  # target attention layers.
                      "gate_proj", "up_proj", "down_proj",],   # also target feed forward layers for better performance.
    lora_alpha = 16, # 16 is a good scaling factor for the LoRA updates. not too high nor too low.
    lora_dropout = 0, # no dropout needed for our use case.
    bias = "none", # we won't train the bias terms, only the LoRA adapters.
    use_gradient_checkpointing = "unsloth", # saving memory via unsloth.
)


dataset = load_dataset("json", data_files="schema_dataset.jsonl", split="train") # (instruction, input, output) format dataset for training.

def formatting_prompts_func(examples): # Convert each example into a single prompt string that includes the instruction, input, and output in a structured format. This is what the model will see during training.
    instructions = examples["instruction"] # assigning the instruction, input and output fields.
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = [] # we will store the formatted prompt strings here.
    
    for instruction, input_text, output in zip(instructions, inputs, outputs): # for loop to format each example into a prompt string.
        
        text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output} <|endoftext|>" 
        texts.append(text) # append the formatted prompt to the list. 
    return { "text" : texts }

mapped_dataset = dataset.map(formatting_prompts_func, batched = True) # apply the format to the entire dataset. 


trainer = SFTTrainer(   # training the lora adapters.
    model = model, # the base model with LoRA adapters.
    tokenizer = tokenizer, # the tokenizer for encoding the prompts.
    train_dataset = mapped_dataset, # the formatted dataset with instruction, input and output as text.
    dataset_text_field = "text",   # the field in the dataset that contains the text prompts.   
    max_seq_length = max_seq_length, # the max sequence length for training.
    dataset_num_proc = 2, # 2 cpu cores for processing the dataset. can adjust later if needed.
    packing = False, # no packing of sequences, we will train on individual prompts.
    args = TrainingArguments( 
        per_device_train_batch_size = 1, # batch size of 1 is good for my GPU.
        gradient_accumulation_steps = 4, 
        warmup_steps = 5, # warmup first then start the training. helps with convergence.
        max_steps = 60, # 60 epochs is a good starting point for training. we can adjust based on performance.
        learning_rate = 2e-4, # not too high nor too low. good for fine tuning.
        fp16 = not torch.cuda.is_bf16_supported(), # try bf16 if not supported then use fp16.
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1, # log all the steps.
        optim = "adamw_8bit", # our optimizer
        weight_decay = 0.01, # small weight decay to prevent overfitting.
        lr_scheduler_type = "linear", 
        seed = 3407, 
        output_dir = "outputs",
    ),
)


print("Starting LoRA fine tuning...")
trainer_stats = trainer.train() # calling the training loop.


model.save_pretrained("harmony_sql_lora") # save the fine tuned LoRA adapter. This will save only the adapter weights, not the entire base model, so it's very efficient.
tokenizer.save_pretrained("harmony_sql_lora") # save the tokenizer as well, so we can use it later for inference. The tokenizer is usually small and doesn't take much space.
print("Training complete! LoRA adapter saved to /harmony_sql_lora")