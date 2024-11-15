# Standard libraries
import concurrent.futures
import logging
import os
import threading

# Non-standard libraries
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt

# Custom libraries
import config
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
class ChatGPTEvaluator:
    """
    ChatGPTEvaluator class.

    Notes
    -----
    Used to evaluate LLM responses via the OpenAI Chat Completion API
    """

    def __init__(self, model=DEFAULT_MODEL, save_dir=None):
        """
        Initialize the ChatGPTEvaluator class.

        Parameters
        ----------
        model : str, optional
            The OpenAI model to be used for evaluation, by default DEFAULT_MODEL.
        save_dir : str, optional
            The directory to save evaluation results. Defaults to a directory
            within config.DIR_EVALUATIONS based on the model name.
        """
        self.model = model
        self.save_dir = save_dir or os.path.join(config.DIR_EVALUATIONS, model)
        self.max_worker = config.MAX_WORKER_AUTOEVAL
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)


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
        llm_input_col="res",
        llm_response_col="eval_res",
    ):
        """
        Evaluate a dataset using the OpenAI API.

        Parameters
        ----------
        data : list of dict
            Each dict contains a LLM-generated response to a unique prompt to
            evaluate
        task : str
            Name of the task to evaluate.
        resume : bool, optional
            If True, then try to resume evaluation from a saved progress file
            with the same filename as `progress_filename`. Default is True.
        progress_filename : str, optional
            Filename for saving or resuming progress. Default is
            `eval_progress.json`.
        llm_input_col : str, optional
            Key to LLM response from initial prompt to evaluate. Overwrites "res"
            in config.config prompts
        llm_response_col : str, optional
            Key to store the judge LLM's response.

        Returns
        -------
        list
            The evaluated data.
        """
        def save_progress_callback(future):
            if future.exception() is not None:
                LOGGER.error("An error occurred: %s", str(future.exception()))
                self.save_progress(data, filename=progress_filename)

        def process_row(item, row):
            try:
                if not row.get(llm_response_col):
                    llm_response = openai_chat_completion(item, model=self.model)
                    row[llm_response_col] = llm_response
            except Exception as error_msg:
                raise error_msg

        # Early return, if no data provided
        if not data:
            return []

        # Prepare prompts for evaluating LLM responses
        prompts = prepare_llm_eval_prompts(data, task, llm_input_col)

        # If specified, resume from previous evaluation
        if resume:
            load_path = os.path.join(self.save_dir, progress_filename)
            try:
                prev_data = json_utils.load_json(load_path)
                if prev_data and len(data) == len(prev_data):
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
            futures = [executor.submit(process_row, item, row) for item, row in zip(prompts, data)]

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


class ChatGPTGenerator:
    """
    ChatGPTGenerator class.

    Notes
    -----
    Used to perform generation via the OpenAI Chat Completion API
    """

    def __init__(self, model=DEFAULT_MODEL, save_dir=None):
        """
        Initialize the ChatGPTGenerator class.

        Parameters
        ----------
        model : str, optional
            The OpenAI model to be used for evaluation, by default DEFAULT_MODEL.
        save_dir : str, optional
            The directory to save evaluation results. Defaults to a directory
            within config.DIR_GENERATIONS based on the model name.
        """
        self.model = model
        self.save_dir = save_dir or os.path.join(config.DIR_GENERATIONS, model)
        self.max_worker = config.MAX_WORKER_AUTOEVAL
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def save_progress(self, data, filename="chatgpt_gen.json", **save_kwargs):
        """
        Save progress to a JSON file.

        Parameters
        ----------
            data : list of dict
                Data to be saved.
            filename : str
                Filename to save date
        """
        save_path = os.path.join(self.save_dir, filename)
        json_utils.save_json(data, save_path, **save_kwargs)


    def infer(
        self, data,
        resume=True, progress_filename="infer_progress.json",
        llm_input_col="prompt", llm_response_col="res",
    ):
        """
        Perform inference on a dataset using the OpenAI API.

        Parameters
        ----------
        data : list of dict
            Each dict contains a prompt in the `llm_input_col` to perform
            inference on.
        resume : bool, optional
            If True, then try to resume inference from a saved progress file
            with the same filename as `progress_filename`. Default is True.
        progress_filename : str, optional
            Filename for saving or resuming progress.
        llm_input_col : str, optional
            Key to prompt to perform inference on
        llm_response_col : str, optional
            Key to store LLM's response.

        Returns
        -------
        list
            The evaluated data.
        """
        def save_progress_callback(future):
            if future.exception() is not None:
                LOGGER.error("An error occurred: %s", str(future.exception()))
                self.save_progress(data, filename=progress_filename)

        def process_row(item, row):
            try:
                if not row.get(llm_response_col):
                    llm_response = openai_chat_completion(item, model=self.model)
                    row[llm_response_col] = llm_response
            except Exception as error_msg:
                raise error_msg

        # Early return, if no data provided
        if not data:
            return []

        # Ensure all rows have a prompt
        assert all(llm_input_col in row for row in data), "All rows must have a prompt specified!"

        # Assume full prompt is specified in the llm input column
        prompts = [data.get(llm_input_col) for data in data]

        # If specified, resume from previous inference
        if resume:
            load_path = os.path.join(self.save_dir, progress_filename)
            try:
                prev_data = json_utils.load_json(load_path)
                if prev_data and len(data) == len(prev_data):
                    LOGGER.info("Resuming inference from saved progress.")
                    data = prev_data
            except FileNotFoundError:
                LOGGER.warning("No saved progress file found at %s. Starting a new inference.", load_path)

        # Perform input sanitization
        assert isinstance(data, list), f"Data must be a list. data={data}"
        assert data, "Data provided is empty!"

        # Create thread lock
        lock = threading.Lock()

        # Perform LLM generation requests in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_worker) as executor:
            futures = [executor.submit(process_row, item, row) for item, row in zip(prompts, data)]

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


################################################################################
#                               Helper Functions                               #
################################################################################
@retry(wait=wait_random_exponential(min=1, max=120), stop=stop_after_attempt(6))
def openai_chat_completion(text_or_msgs=None, model=DEFAULT_MODEL, temperature=0):
    """
    Sends string/messages from the OpenAI ChatCompletion API.

    Parameters
    ----------
    text_or_msgs : str or list of dict, optional
        The input user text to be processed by the API, or a list of messages
        to be processed by the API.
    model : str, optional
        The model to use for the API request. Default is "gpt-4o-2024-08-06".
    temperature : float, optional
        The temperature to use for the API request. Default is 0.

    Returns
    -------
    Union[None, str]
        If the API response is null or an empty string, returns None.
        Otherwise, returns the response from the API.
    """
    assert text_or_msgs, f"Please provide valid input text/messages! Received: `{text_or_msgs}`"
    assert isinstance(text_or_msgs, (str, list)), f"Input text/messages must be either a str or a List[Dict]!"
    try:
        # Prepare input
        messages = text_or_msgs
        if isinstance(text_or_msgs, str):
            messages = [{"role": "user", "content": text_or_msgs}]
        assert isinstance(messages, list)

        # Configure client
        api_key = config.OPENAI_KEY
        if config.OPENAI_API_URL is not None:
            client = OpenAI(
                api_key=api_key,
                base_url=config.OPENAI_API_URL
            )
        else:
            client = OpenAI(api_key=api_key)

        # Send request to chat completions API
        # Temperature will be set to 0 for deterministic output (i.e., greedy decoding)
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        if not stream.choices[0].message.content:
            raise ValueError("The response from the API is NULL or an empty string!")
        response = stream.choices[0].message.content
    except Exception as e:
        print(e)
        return None
    return response


def prepare_llm_eval_prompts(data, task=None, llm_input_col="res"):
    """
    Prepare evaluation prompts for a given task using LLM-generated responses.

    This function formats prompts for language model evaluation by either 
    filling in placeholders within a template prompt with data from each row 
    or by appending the LLM response to a fixed prompt template.

    Parameters
    ----------
    data : list of dict
        The dataset containing LLM-generated responses to be evaluated.
    task : str, optional
        The name of the task for which prompts need to be prepared. This is 
        used to fetch the corresponding prompt template and mappings from 
        the configuration.
    llm_input_col : str, optional
        The column name in the data dict that contains the input text for 
        the LLM. Defaults to "res".

    Returns
    -------
    list of str
        A list of formatted prompts ready for evaluation.
    """
    # Set up prompt formatters
    # If prompt contains row formatters, then fill them in with row information
    task_prompt_dict = config.TASK_TO_PROMPT_DICT.get(task, {})
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
                # CASE 1: If "res" was specified, but LLM input column is different
                #         then convert
                if v == "res" and llm_input_col != "res":
                    val = row[llm_input_col]
                # CASE 2: Any other column
                else:
                    val = row[v]
                single_prompt = single_prompt.replace(k, str(val))
            prompts.append(single_prompt)
    # CASE 2: Otherwise, simply append LLM response to end of prompt
    else:
        LOGGER.debug("[ChatGPT Evaluator] Concatenating LLM response to prompt")
        prompt = task_prompt_dict.get('prompt', '')
        prompts = [prompt + item[llm_input_col] for item in data]

    return prompts
