import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig
from huggingface_hub import HfApi

# INPUT
DATASET_PATH = r"C:\Users\vipul\Downloads\SBAcase.11.13.17.csv"
HF_TOKEN = "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"
HF_USERNAME = "keva1ya"

def format_row(row):
    prompt = f"""[INST] Assess SBA loan default risk.
Term: {row['Term']} months
Employees: {row['NoEmp']}
Disbursement Gross: ${row['DisbursementGross']}
Gross Approved: ${row['GrAppv']} [/INST] """
    if row['Default'] == 1:
        response = '"High risk of default. Loan is likely to be charged off."'
    else:
        response = '"Low risk of default. Loan is likely to be paid in full."'
    
    return prompt + response + "</s>"

class EpochProgressCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        loss = "N/A"
        # Find the most recent loss logged
        for log in reversed(state.log_history):
            if "loss" in log:
                loss = log["loss"]
                break
        print(f"\n[INFO] --- Epoch {int(state.epoch)}/{args.num_train_epochs} completed! Current training loss: {loss} ---")

def main():
    print("PHASE 2 — LOAD AND PREPROCESS DATASET")
    df = pd.read_csv(DATASET_PATH)
    columns_to_keep = [
        "Term",
        "NoEmp",
        "DisbursementGross",
        "GrAppv",
        "Default"
    ]
    df = df[columns_to_keep]
    df = df.dropna()
    df = df.head(2000)

    print("PHASE 3 — CONVERT TO MISTRAL CHAT FORMAT")
    df["text"] = df.apply(format_row, axis=1)
    dataset = Dataset.from_pandas(df[["text"]])

    print("PHASE 4 — LOAD MODEL (QLORA)")
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN,
    )
    
    print("PHASE 5 — TOKENIZER SETUP")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=HF_TOKEN)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    
    print("PHASE 6 — APPLY LORA")
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    
    print("PHASE 7 — TRAINING CONFIGURATION")
    training_args = SFTConfig(
        output_dir="./mistral-risk-lora-checkpoints",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=8,
        bf16=True,
        logging_steps=10,
        save_steps=100,
        optim="paged_adamw_8bit",
        report_to="none",
        gradient_checkpointing=True,
        dataset_text_field="text",
    )
    
    print("PHASE 8 — TRAIN MODEL")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        callbacks=[EpochProgressCallback()],
    )
    
    train_result = trainer.train()
    
    print("PHASE 9 — SAVE LORA MODEL")
    trainer.model.save_pretrained("./mistral-risk-lora")
    tokenizer.save_pretrained("./mistral-risk-lora")
    
    print("PHASE 10 — MERGE MODEL (CRITICAL)")
    # Clear memory
    del model
    del trainer
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    # Reload base model in fp16 (or default precision) to merge
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        token=HF_TOKEN,
    )
    # Load LoRA adapter
    model_to_merge = PeftModel.from_pretrained(base_model, "./mistral-risk-lora")
    
    # Merge and unload
    merged_model = model_to_merge.merge_and_unload()
    
    # Save merged model
    merged_model.save_pretrained("./mistral-risk-merged")
    tokenizer.save_pretrained("./mistral-risk-merged")
    
    print("PHASE 11 — UPLOAD TO HUGGING FACE")
    api = HfApi()
    repo_id = f"{HF_USERNAME}/mistral-risk-model"
    
    try:
        api.create_repo(repo_id=repo_id, exist_ok=True, token=HF_TOKEN)
        api.upload_folder(
            folder_path="./mistral-risk-merged",
            repo_id=repo_id,
            repo_type="model",
            token=HF_TOKEN,
        )
        repo_link = f"https://huggingface.co/{repo_id}"
    except Exception as e:
        repo_link = f"Failed to upload: {e}"

    print("\n--- PHASE 12 — OUTPUT REQUIREMENTS ---")
    print(f"Final training loss: {train_result.training_loss}")
    print(f"Number of samples processed: {len(dataset) * training_args.num_train_epochs}")
    print(f"Saved model path: ./mistral-risk-merged")
    print(f"Hugging Face repo link: {repo_link}")

if __name__ == "__main__":
    main()
