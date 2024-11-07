# Standard libraries
import logging
import os

# Non-standard libraries
import torch
from tqdm import tqdm
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# Custom libraries
import src.config as config
from src.utils import json_utils

PROGRESS_FILENAME = 'longformer_eval.json'

# TODO: Refine this class for more general HF model evaluation

class HuggingFaceEvaluator:
    """
    A class for evaluating text using a Hugging Face model.
    """
    def __init__(self, model_name='LibrAI/longformer-harmful-ro', device=None, save_dir=config.DIR_EVALUATIONS):
        """
        Initialize the HuggingFaceEvaluator class.

        Args:
            model_name (str): Name of the pre-trained model to use.
            device (str): The device to run the model on (e.g., 'cuda:0').
            save_dir (str): Directory to save the evaluation results.
        """
        device=config.device
        if device==None:
            self.device='cpu' if torch.cuda.is_available() else 'cuda'
        else:
            self.device=device
        # Load the model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.classifier = pipeline('text-classification', model=model, tokenizer=tokenizer, device=self.device)
        

        self.save_dir = save_dir
        # Create save directory if it doesn't exist
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def evaluate(self, data, resume=False, progress_filename=PROGRESS_FILENAME):
        """
        Evaluate a dataset using the loaded Hugging Face model.

        Args:
            data: List of data to be evaluated.
            resume (bool): Whether to resume from a saved progress file. Default is False.
            progress_filename (str): Filename for saving or resuming progress.

        Returns:
            List: The evaluated data.
        """
        if resume:
            load_path = os.path.join(self.save_dir, progress_filename)
            try:
                data = json_utils.load_json(load_path)
                logging.info("Resuming evaluation from saved progress.")
            except FileNotFoundError:
                logging.warning("No saved progress file found at %s. Starting a new evaluation.", load_path)

        assert isinstance(data, list), "Data must be a list."
        logging.info('Total data number: {}'.format(len(data)))
        # Filter out invalid data
        data = [row for row in data if row['res'] is not None and row != ""]
        logging.info('Total data number: %d', len(data))
        logging.info('Evaluating...')

        evaluated_data = self._evaluate_batch(data)

        json_utils.save_json(data, os.path.join(self.save_dir, progress_filename))
        return evaluated_data


    def _evaluate_batch(self,data, batch_size=32, progress_filename=PROGRESS_FILENAME):
        results = []
        for i in tqdm(range(0, len(data), batch_size)):
            batch = data[i:i+batch_size]
            texts = [item.get('res', '') for item in batch if 'eval_res' not in item]
            if texts:
                try:
                    batch_outputs = self.classifier(texts)
                    assert len(batch_outputs) == len(texts)
                    idx = 0
                    for item in batch:
                        if 'eval_res' not in item:
                            item['eval_res'] = batch_outputs[idx]["label"]
                            idx += 1
                    results.extend(batch)
                    logging.info("Processed batch from %s to %s", i, i+batch_size)
                except Exception as e:
                    logging.error("Error processing batch %s to %s: %s", i, i+batch_size, str(e))
                    json_utils.save_json(data, os.path.join(self.save_dir, progress_filename))
                    raise
            else:
                results.extend(batch)
        return results