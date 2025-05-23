import argparse
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import gdown
from transformers import BlipProcessor, BlipForQuestionAnswering
from peft import PeftModel

def download_model_weights(output_dir, google_drive_url):
    """Download model weights from Google Drive if not already present."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Downloading model weights from Google Drive to {output_dir}...")
        gdown.download_folder(url=google_drive_url, output=output_dir, quiet=False)
    else:
        print(f"Model weights already exist at {output_dir}")

def load_finetuned_model(model_dir):
    """Load the fine-tuned BLIP model with LoRA adapter."""
    base_model_name = "Salesforce/blip-vqa-base"
    device = torch.device("cpu")

    # Load processor
    processor = BlipProcessor.from_pretrained(model_dir, local_files_only=True)

    # Load the base BLIP model
    base_model = BlipForQuestionAnswering.from_pretrained(base_model_name)

    # Load the LoRA adapter and apply it to the base model
    model = PeftModel.from_pretrained(base_model, model_dir, is_trainable=False, local_files_only=True)

    # Move model to device
    model.to(device)
    model.eval()
    return processor, model, device

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help='Path to image folder')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to metadata CSV')
    args = parser.parse_args()

    # Google Drive URL for model weights
    GOOGLE_DRIVE_URL = "https://drive.google.com/drive/folders/1IUud4PUPf1WAlW5wFNp2TGZJ-bEeSYsK?usp=sharing"
    MODEL_DIR = "./blip_lora_finetuned"

    # Download model weights
    download_model_weights(MODEL_DIR, GOOGLE_DRIVE_URL)

    # Load model and processor
    print("Loading the model BLIP with LoRA...")
    processor, model, device = load_finetuned_model(MODEL_DIR)

    # Load metadata CSV
    df = pd.read_csv(args.csv_path)

    generated_answers = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_path = os.path.join(args.image_dir, row['image_name'])
        question = str(row['question'])
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, text=question, return_tensors="pt").to(device)
            with torch.no_grad():
                # using kv-cache for faster inferencing
                generated_ids = model.generate(**inputs, use_cache=True)
                answer = processor.decode(generated_ids[0], skip_special_tokens=True)
            # Post-process answer to be one word and lowercase
            answer = str(answer).split()[0].lower()
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            answer = "error"
        generated_answers.append(answer)

    # Add generated answers to DataFrame and save to results.csv
    df["generated_answer"] = generated_answers
    df.to_csv("results.csv", index=False)
    print("Inference complete. Results saved to results.csv")

if __name__ == "__main__":
    main()