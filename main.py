import argparse
from src.train import train_with_cv
from src.predict import load_model_and_tokenizer, predict_resume_topk
from src.logger import setup_logger

logger = setup_logger("MainLogger")

def run_training():
    logger.info("Starting training with cross-validation...")
    train_with_cv(
        data_path="data/Resume.csv",
        model_name="bert-base-uncased",
        output_dir="models",
        num_folds=5
    )
    logger.info("Training completed.")

def run_transformer_inference(text):
    logger.info("Running inference using Transformer model...")
    model_dir = "models/fold_1"
    label_path = "outputs/label_classes.json"

    model, tokenizer, label_encoder = load_model_and_tokenizer(model_dir, label_path)
    results = predict_resume_topk(text, model, tokenizer, label_encoder, top_k=5)

    logger.info("Inference Results (Transformer):")
    for item in results:
        logger.info(f"{item['job_title']}: {item['percentage']:.2f}%")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume Classification")
    parser.add_argument("--mode", choices=["train", "predict"], required=True, help="Run mode: train or predict")
    parser.add_argument("--text", type=str, help="Resume text for prediction (required if mode=predict)")

    args = parser.parse_args()

    if args.mode == "train":
        run_training()

    elif args.mode == "predict":
        if not args.text:
            logger.error("You must provide --text for prediction.")
        else:
            results = run_transformer_inference(args.text)
            for r in results:
                print(f"{r['job_title']}: {r['percentage']:.2f}%")