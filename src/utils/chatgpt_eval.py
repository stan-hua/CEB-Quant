# Standard libraries
import concurrent.futures
import logging
import os
import threading

# Non-standard libraries
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt

# Custom libraries
from src.config import config
from src.utils import json_utils


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

        api_key = config.OPENAI_KEY
        if config.OPENAI_API_URL is not None:
            client = OpenAI(
                api_key=api_key,
                base_url=config.OPENAI_API_URL
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
        self.max_worker = config.MAX_WORKER_AUTOEVAL
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        # openai.api_key = config.openai_key


    def save_progress(self, data, filename='auto_eval.json', **save_kwargs):
        """
        Save evaluation progress to a JSON file.

        Args:
            data: Data to be saved.
            filename (str): Name of the file for saving the data.
        """
        save_path = os.path.join(self.save_dir, filename)
        json_utils.save_json(data, save_path, **save_kwargs)


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


        task_to_prompt = config.TASK_TO_PROMPT_DICT
        # If prompt contains row formatters, then fill them in with row information
        task_prompt_dict = task_to_prompt.get(task, {})
        use_prompt_formatter = "mapping" in task_prompt_dict

        # Prepare prompts
        # CASE 1: Prompt contains string formatters
        prompts = []
        if use_prompt_formatter:
            replace_dict = task_prompt_dict.get('mapping', {})
            prompt = task_prompt_dict.get('prompt', '')
            for row in data:
                single_prompt = prompt
                for k, v in replace_dict.items():
                    single_prompt = single_prompt.replace(k, str(row[v]))
                prompts.append(single_prompt)
        # CASE 2: Otherwise, simply append LLM response to end of prompt
        else:
            LOGGER.debug("[ChatGPT Evaluator] Concatenating LLM response to prompt")
            prompt = task_prompt_dict.get('prompt', '')
            prompts = [prompt + item['res'] for item in data]

        # If specified, resume from previous evaluation
        if resume:
            load_path = os.path.join(self.save_dir, progress_filename)
            try:
                prev_data = json_utils.load_json(load_path)
                if prev_data:
                    LOGGER.info("Resuming evaluation from saved progress.")
                    data = prev_data
            except FileNotFoundError:
                LOGGER.warning("No saved progress file found at %s. Starting a new evaluation.", load_path)

        # Perform input sanitization
        assert isinstance(data, list), f"Data must be a list. data={data}"
        assert data, "Data provided is empty!"
        assert task is not None, "Task must be specified for evaluation."

        # Create thread lock
        lock = threading.Lock()

        # Perform LLM generation requests in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_worker) as executor:
            futures = [executor.submit(process_item, item, row) for item, row in zip(prompts, data)]

            # Add a callback to handle completion and errors
            for idx, future in enumerate(concurrent.futures.as_completed(futures)):
                future.add_done_callback(save_progress_callback)
                if idx % 10 == 0:
                    self.save_progress(data, filename=progress_filename, lock=lock)

            # Wait for all futures to complete
            concurrent.futures.wait(futures)

        # Save progress
        self.save_progress(data, filename=progress_filename)

        return data
