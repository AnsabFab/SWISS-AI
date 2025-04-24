import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# Paths
MODEL_PATH = "phi2-legal-finetuned"  # Path to your fine-tuned model

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# For LoRA fine-tuned model
if "peft_config.json" in MODEL_PATH:
    # Load the base model and tokenizer
    base_model_path = "/data/models_weights/phi-2"  # Path to original Phi-2 model
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    # Load configuration to get base model path
    config = PeftConfig.from_pretrained(MODEL_PATH)
    
    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True,
        device_map="auto"
    )
    
    # Load the LoRA adapter
    model = PeftModel.from_pretrained(model, MODEL_PATH)
else:
    # For full fine-tuned model (non-LoRA)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True,
        device_map="auto"
    )

# Set the model to evaluation mode
model.eval()

def generate_response(instruction, context="", max_length=512, temperature=0.7, top_p=0.9):
    """
    Generate a response using the fine-tuned model
    
    Args:
        instruction (str): The instruction for the model
        context (str): Optional context to provide
        max_length (int): Maximum length of the generated text
        temperature (float): Sampling temperature (higher = more random)
        top_p (float): Nucleus sampling parameter
        
    Returns:
        str: The generated response
    """
    # Format the prompt in the same way as during training
    prompt = f"""### Instruction:
{instruction}
### Context:
{context}
### Response:
"""
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate a response
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the response part (after "### Response:")
    response_part = response.split("### Response:")[-1].strip()
    
    return response_part

# Example usage
if __name__ == "__main__":
    instruction = "Explain the concept of force majeure in contract law."
    context = "I'm reviewing a commercial contract that contains a force majeure clause."
    
    print("Generating response...")
    response = generate_response(instruction, context)
    print("\n" + "="*50)
    print("Instruction:", instruction)
    print("Context:", context)
    print("="*50)
    print("Response:")
    print(response)
    print("="*50)
    
    # Interactive mode
    print("\nEnter 'q' to quit at any time.")
    while True:
        instruction = input("\nEnter instruction: ")
        if instruction.lower() == 'q':
            break
        
        context = input("Enter context (optional): ")
        if context.lower() == 'q':
            break
        
        response = generate_response(instruction, context)
        print("\nResponse:")
        print(response)