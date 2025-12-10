import torch
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# --- Configuration ---
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
SEED = 42

def load_model_and_tokenizer():
    """Loads the Qwen model and tokenizer, handling device mapping."""
    
    print(f"Loading model: {MODEL_ID}...")
    
    # Determine device and data type for efficient loading
    if torch.cuda.is_available():
        device = "cuda"
        # Use bfloat16 if supported for better performance/memory, otherwise fall back to float16 or float32
        try:
            dtype = torch.bfloat16
        except AttributeError:
            dtype = torch.float16
        print(f"Found CUDA device. Using {device} with dtype {dtype}.")
        
    else:
        device = "cpu"
        dtype = torch.float32
        print(f"No CUDA device found. Using CPU (this will be very slow).")


    # Load the model and tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        # Load the model with device mapping and data type optimization
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            device_map="auto" # Automatically handles loading model layers across devices
        )
        model.eval() # Set model to evaluation mode

        # Create a text-generation pipeline that wraps the model+tokenizer.
        # The pipeline exposes a simple call interface and will use the model we loaded.
        device_id = 0 if device == "cuda" else -1
        gen_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )

        # Set deterministic flags where possible
        torch.manual_seed(SEED)
        if device == "cuda":
            torch.cuda.manual_seed_all(SEED)
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:
                pass

        return gen_pipe, tokenizer, device
    
    except Exception as e:
        print("\n--- ERROR ---")
        print("Failed to load the model or tokenizer.")
        print("Please ensure you have the following packages installed:")
        print("pip install transformers torch accelerate")
        print(f"Original error: {e}")
        sys.exit(1)


def chat_cli(gen_pipe, tokenizer, device):
    """Starts the interactive command-line chat session."""
    
    print("\n--- Qwen 2.5 3B CLI Chat ---")
    print("Model loaded successfully. Start chatting!")
    print("Type 'exit' or 'quit' to stop the session.")
    print("-" * 30)

    # Initialize the conversation history with a system prompt
    messages = [
        {"role": "system", "content": "You are an assistant tasked with giving probability estimates for user queries."}
    ]

    try:
        while True:
            # Get user input
            # user_input = input("You: ").strip()
            user_input = "Who were key players for Liverpool Football Club in 2018?"
            if user_input.lower() in ["quit", "exit"]:
                print("Exiting chat. Goodbye!")
                break
            
            if not user_input:
                continue

            # Add user message to history
            messages.append({"role": "user", "content": user_input})
            
            # 1. Apply the Qwen chat template to the messages
            # This converts the list of dicts into the model's required input string format.
            input_text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )

            # 2. Use the pipeline to generate the response (deterministic)
            print("Assistant: ", end="", flush=True)

            # Call the pipeline. Use greedy decoding (do_sample=False) and a fixed seed.
            outputs = gen_pipe(
                input_text,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.0,
                return_full_text=False,
            )

            # The pipeline returns a list of dicts with 'generated_text'
            generated_text = outputs[0]["generated_text"] if outputs else ""
            
            # Print the final response and add it to the message history
            print(generated_text)
            messages.append({"role": "assistant", "content": generated_text})

    except KeyboardInterrupt:
        print("\nExiting chat. Goodbye!")


if __name__ == "__main__":
    # Check for necessary libraries before starting
    try:
        import torch
        import transformers
    except ImportError:
        print("Required libraries missing. Please run:")
        print("pip install transformers torch accelerate")
        sys.exit(1)

    pipe, tokenizer, device = load_model_and_tokenizer()
    chat_cli(pipe, tokenizer, device)