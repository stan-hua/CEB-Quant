# Standard libraries
import concurrent.futures
import logging
import os

# Non-standard libraries
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt

# Custom libraries
from src.config import config
from utils import json_utils


################################################################################
#                                  Constants                                   #
################################################################################
LOGGER = logging.getLogger(__name__)

# Default OpenAI model for evaluation
DEFAULT_MODEL = "gpt-4o-2024-08-06"


################################################################################
#                                   Classes                                    #
################################################################################
@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(6))
def get_res(string, model=DEFAULT_MODEL, temperature=0, message=None):
    """
    Retrieve a response from the OpenAI ChatCompletion API.

    Args:
        string (str): The input string to process.
        model (str): The model to use for generating the response. Default is DEFAULT_MODEL.
        temp (float): The temperature setting for the API request. Default is 0 for deterministic output.

    Returns:
        str: The API response content.

    Raises:
        ValueError: If the API response is null or an empty string.
    """
    try:
        if message is None:
            message = [{"role": "user", "content": string}]

        api_key = config.openai_key
        if config.openai_api_base is not None:
            client = OpenAI(
                api_key=api_key,
                base_url=config.openai_api_base
            )
        else:
            client = OpenAI(api_key=api_key)
        # Temperature will be set to 0 for deterministic output (i.e., greedy decoding)
        stream = client.chat.completions.create(model=model,
                                                messages=message,
                                                temperature=temperature,)
        if not stream.choices[0].message.content:
            raise ValueError("The response from the API is NULL or an empty string!")
        response = stream.choices[0].message.content
    except Exception as e:
        print(e)
        return None
    return response


class ChatGPTEvaluator:
    """
    A class for automating the evaluation of text using the OpenAI API.
    """

    def __init__(self, model=DEFAULT_MODEL, save_dir='saved_evaluations'):
        """
        Initialize the AutoEvaluator class.

        Args:
            save_dir (str): Directory for saving evaluation results.
        """
        self.model = model
        self.save_dir = save_dir
        self.max_worker = config.max_worker_auto_eval
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        # openai.api_key = config.openai_key


    def save_progress(self, data, filename='auto_eval.json'):
        """
        Save evaluation progress to a JSON file.

        Args:
            data: Data to be saved.
            filename (str): Name of the file for saving the data.
        """
        save_path = os.path.join(self.save_dir, filename)
        json_utils.save_json(data, save_path)
        LOGGER.info("Progress saved to %s", save_path)


    def evaluate(
        self, data, task,
        resume=True,
        progress_filename="eval_progress.json",
        llm_response_col="eval_res",
    ):
        def save_progress_callback(future):
            if future.exception() is not None:
                LOGGER.error("An error occurred: %s", str(future.exception()))
                self.save_progress(data, filename=progress_filename)

        def process_item(item, row):
            try:
                if llm_response_col not in row:
                    llm_response = get_res(item, model=self.model)
                    row[llm_response_col] = llm_response
            except Exception as error_msg:
                raise error_msg


        task_to_prompt = config.task_prompt
        # If prompt contains row formatters, then fill them in with row information
        use_prompt_formatter = "mapping" in task_to_prompt.get(task, {})

        # Prepare prompts
        # CASE 1: Prompt contains string formatters
        prompts = []
        if use_prompt_formatter:
            replace_dict = task_to_prompt.get(task, {}).get('mapping', {})
            prompt = task_to_prompt.get(task, {}).get('prompt', '')
            for row in data:
                single_prompt = prompt
                for k, v in replace_dict.items():
                    single_prompt = single_prompt.replace(k, str(row[v]))
                prompts.append(single_prompt)
        # CASE 2: Otherwise, simply append LLM response to end of prompt
        else:
            prompt = task_to_prompt.get(task, {}).get('prompt', '')
            prompts = [prompt + item['res'] for item in data]

        # If specified, resume from previous evaluation
        if resume:
            load_path = os.path.join(self.save_dir, progress_filename)
            try:
                data = json_utils.load_json(load_path)
                LOGGER.info("Resuming evaluation from saved progress.")
            except FileNotFoundError:
                LOGGER.warning("No saved progress file found at %s. Starting a new evaluation.", load_path)

        assert isinstance(data, list), "Data must be a list."
        assert task is not None, "Task must be specified for evaluation."

        LOGGER.info('Total data number: %d', len(data))
        LOGGER.info('Evaluating...')

        # Perform LLM generation requests in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_worker) as executor:
            futures = [executor.submit(process_item, item, row) for item, row in zip(prompts, data)]

            # Add a callback to handle completion and errors
            for future in concurrent.futures.as_completed(futures):
                future.add_done_callback(save_progress_callback)

            # Wait for all futures to complete
            concurrent.futures.wait(futures)

        self.save_progress(data, filename=progress_filename)
        return data
