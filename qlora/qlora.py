import os
import evaluate
import numpy as np
import torch

from datasets import load_dataset
from evaluate import load

from transformers import (
    BertTokenizerFast,
    BertForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
    pipeline,
    set_seed,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)

base_model = "bert-base-uncased"
base_dataset = "squad"
tokenizer = BertTokenizerFast.from_pretrained(base_model)

output_dir = "./results"
logging_dir = "./logs"
trained_model_name = "bert-qlora-squad"
trained_model_path = f"./{trained_model_name}"

# Set seed for reproducibility
# This is important for consistent results across runs
# Seed will be used as static variable for all tests so that results are comparable
seeds = [42, 1234, 2023, 2024, 2025]
final_results = []

# 1. Preprocessing for SQuAD
# --------------------------------
# Load the dataset and preprocess it for SQuAD

def preprocess_fn(ex):
    # tokenize with offset mapping to align char -> token positions
    tok = tokenizer(
        ex["question"], ex["context"],
        truncation="only_second",
        max_length=384,
        stride=128,
        return_overflowing_tokens=False,
        return_offsets_mapping=True,
        padding="max_length",
    )
    offsets = tok.pop("offset_mapping")
    start_char = ex["answers"]["answer_start"][0]
    end_char = start_char + len(ex["answers"]["text"][0])
    # map to token indices
    start_idx = end_idx = 0
    for i, (s, e) in enumerate(offsets):
        if s <= start_char < e:
            start_idx = i
        if s < end_char <= e:
            end_idx = i
            break
    tok["start_positions"] = start_idx
    tok["end_positions"] = end_idx
    return tok

# load and preprocess
raw = load_dataset(base_dataset)
train_ds = raw["train"].map(preprocess_fn, batched=False)
val_ds   = raw["validation"].map(preprocess_fn, batched=False)
# set torch format
train_ds.set_format(type="torch", columns=["input_ids","attention_mask","start_positions","end_positions"])
val_ds.set_format(type="torch", columns=["input_ids","attention_mask","start_positions","end_positions"])


# Prepare evaluation metrics

def evaluate_qa_model(model_path, dataset_name=base_dataset, split="validation", tokenizer_path=None, base_model_name=base_model):
    """
    Evaluate a QA model using Exact Match and F1 Score with batch processing.

    Parameters:
    - model_path (str): Path or HuggingFace Hub ID of the QA model (should be path to LoRA weights).
    - dataset_name (str): Name of the HuggingFace dataset
    - split (str): Dataset split to use (default: "validation").
    - tokenizer_path (str or None): Tokenizer path if different from model_path.
    - base_model): (str): Name of the original base model.

    Returns:
    - dict: Contains 'exact_match' and 'f1' scores.
    """
    # Load dataset
    dataset = load_dataset(dataset_name, split=split)

    # Load base model
    base_model = BertForQuestionAnswering.from_pretrained(base_model_name)

    # Load LoRA weights and apply to base model
    model = PeftModel.from_pretrained(base_model, model_path)

    # Load tokenizer
    tokenizer_path = tokenizer_path or model_path
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)

    # Pass the combined model and tokenizer to the pipeline
    # The pipeline will automatically handle batching and GPU usage
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, device=0) # device=0 for GPU

    # Prepare data for the pipeline and get predictions
    pipeline_output = []
    for item in dataset:
        try:
            result = qa_pipeline(question=item["question"], context=item["context"])
            pipeline_output.append(result)
        except Exception as e:
            print(f"Error processing example {item['id']}: {e}")
            # Append a placeholder or handle the error as needed.
            # For evaluation metrics, you might want to skip this example or assign a default.
            # Here, we'll append a placeholder to maintain the list length for referencing.
            pipeline_output.append({"answer": ""}) # Append empty string as prediction

    # Prepare predictions and references in the format expected by evaluate
    predictions = []
    references = []

    for i, item in enumerate(dataset):
        example_id = item["id"]
        gold_answers = item["answers"]["text"]
        gold_answer_starts = item["answers"]["answer_start"]

        # Check if the index i is within the bounds of pipeline_output
        if i < len(pipeline_output):
            predicted_answer = pipeline_output[i]["answer"]
        else:
            predicted_answer = "" # Default empty if there was an error and no prediction

        predictions.append({'id': example_id, 'prediction_text': predicted_answer})
        references.append({'id': example_id, 'answers': {'text': gold_answers, 'answer_start': gold_answer_starts}})


    # Compute metrics
    if not predictions:
        print("No successful predictions were made.")
        return {"exact_match": 0.0, "f1": 0.0}

    squad_metric = evaluate.load(dataset_name)
    results = squad_metric.compute(predictions=predictions, references=references)

    return results


# 2. Initialize Training Arguments
# --------------------------------

for seed in seeds:
    print(f"Running with seed: {seed}")
    set_seed(seed)  # Set the seed for reproducibility

    # Reset GPU memory stats if using GPU
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()  # Reset peak memory stats if using GPU

    # Prepare model for training

    # Setup config for bitsandbytes to make the model 4-bit (QLoRA support)
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Setup model
    model_qlora = BertForQuestionAnswering.from_pretrained(
        base_model,
        quantization_config=bnb_cfg,
        device_map="auto"
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model_qlora)

    # setup lora for quantized model
    lora_cfg_q = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["query","value"],
        lora_dropout=0.1,
        bias="none",
        task_type="QUESTION_ANS",
        use_rslora=False
    )

    #Load model in peft lib
    model_qlora = get_peft_model(model_qlora, lora_cfg_q)

    t_args = TrainingArguments(
        # model output directory
        output_dir=f"./results/seed_{seed}",

        # as in BERT paper for training SQuAD v1.1
        per_device_train_batch_size=32,
        num_train_epochs=3,
        learning_rate=5e-5,

        fp16=True,

        # logging when training config
        logging_strategy="steps",
        logging_steps=500,
        logging_dir=f"./logs/seed_{seed}",

        save_strategy="epoch",
        save_steps=2000,

        eval_steps=2000,
        eval_strategy="epoch",

        gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Use non-reentrant gradient checkpointing
        max_grad_norm=None,  # Gradient clipping to avoid exploding gradients

        report_to="none",

        label_names=["start_positions", "end_positions"]  # Add this line to add label to hidden label by peft
    )

    trainer = Trainer(
        model=model,
        args=t_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=default_data_collator  # to use auto padding
    )

    trainer.train()

    # Check peak memory
    peak_mem = None
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024**3)  # in GB
        print(f"Peak CUDA memory (GB): {peak_mem:.2f}")

    # Save the model
    model.save_pretrained(f"./{trained_model_name}_seed_{seed}")
    tokenizer.save_pretrained(f"./{trained_model_name}_seed_{seed}")
    print(f"Model saved to: ./{trained_model_name}_seed_{seed}")
    # Save the trained model path for evaluation
    trained_model_path = f"./{trained_model_name}_seed_{seed}"
    
    # evaluate the model
    try:
        print("Evaluating model...")
        results = evaluate_qa_model(model_path=trained_model_path, dataset_name=base_dataset, split="validation", tokenizer_path=trained_model_path, base_model_name=base_model)
        print("Results:", results)
    except Exception as e:
        print(f"Error evaluating model: {e}")
        # 3. Train the model with LoRA
        results = None
        trained_model_path = None
        continue
    finally:
        # Clean up resources if needed
        pass

    final_results.append({
        "seed": seed,
        "results": results,
        "trained_model_path": trained_model_path
    })
    # 3. Train the model with LoRA
    print(f"Final results for seed {seed}: {results}")
    # Save final results to a file
    with open(f"./final_results_seed_{seed}.txt", "w") as f:
        f.write(f"Seed: {seed}\n")
        f.write(f"Results: {results}\n")
        f.write(f"Trained Model Path: {trained_model_path}\n")
        if peak_mem is not None:
            f.write(f"Peak CUDA Memory (GB): {peak_mem:.2f}\n")

# summary
print("Final results for all seeds:")
for res in final_results:
    print(f"Seed: {res['seed']}, Results: {res['results']}, Trained Model Path: {res['trained_model_path']}")