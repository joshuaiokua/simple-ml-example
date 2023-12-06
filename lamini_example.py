"""
    Example taken from Lamini.

    Original file is located at
        https://colab.research.google.com/drive/10VFX_wBEZZV7rdySscKL0DCq5Qlpkzjo
"""

import os
import dotenv
import time
import lamini
from llama import QuestionAnswerModel

def download_files_from_google_drive():
    os.system("""
    gdown -q -O "seed_lamini_docs.jsonl" "https://drive.google.com/uc?export=download&id=1SfGp1tVuLTs0WYDugZcxX-EHrmDtYrYJ"
    gdown -q -O "seed_taylor_swift.jsonl" "https://drive.google.com/uc?export=download&id=119sHYYImcXEbGyvS3wWGpkSEVIFdLy6Z"
    gdown -q -O "seed_bts.csv" "https://drive.google.com/uc?export=download&id=1lblhdhKwoiOjlvfk8tr7Ieo4KpvjRm6n"
    gdown -q -O "seed_open_llm.jsonl" "https://drive.google.com/uc?export=download&id=1S7oPPko-UmOr-bqkZ_PREfGKO2f73ZiK"
    """)

def print_training_results(results):
    print("-"*100)
    print("Training Results")
    print(results)
    print("-"*100)

def print_inference(question, finetune_answer, base_answer):
    print('Running Inference for: '+ question)
    print("-"*100)
    print("Finetune model answer: ", finetune_answer)
    print("-"*100)
    print("Base model answer: ", base_answer)
    print("-"*100)

def main():
    # Load the API key
    dotenv.load_dotenv()
    lamini.api_key = os.getenv("LAMINI_API_KEY")

    # Download the seed data
    download_files_from_google_drive()

    # Train the model on the seed data
    finetune_model = QuestionAnswerModel()
    finetune_model.load_question_answer_from_jsonlines("seed_lamini_docs.jsonl")
    finetune_model.config['api_key'] = os.getenv('LAMINI_API_KEY')

    # Train the model
    start=time.time()
    finetune_model.train(enable_peft=True)
    print(f"Time taken: {time.time()-start} seconds")

    # Evaluate base and finetuned models to compare performance
    results = finetune_model.get_eval_results()
    print_training_results(results)

main()