# Standard libraries
import concurrent
import glob
import multiprocessing
import logging
import os
import threading
import time

# Non-standard libraries
import torch
import traceback
import urllib3
from dotenv import load_dotenv
from fastchat.model import load_model
from transformers import AutoTokenizer
from tqdm import tqdm
from vllm import LLM, SamplingParams

# Custom libraries
from src.config.config import MODEL_INFO, ALL_DATASETS
from src.utils import json_utils, llm_gen_utils


load_dotenv()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# Setup logger
LOGGER = logging.getLogger(__name__)

# Configure multiprocessing; to avoid vLLM issues with multi-threading
multiprocessing.set_start_method('spawn')


################################################################################
#                                  Constants                                   #
################################################################################
# Number of threads to use in sending requests to LLM APIs
NUM_WORKERS = 8

# Maximum vLLM model input token length
MAX_MODEL_LEN = 4096

# Default configuration parameters
DEFAULT_CONFIG = {
    "model_provider": "vllm",
    # Local model loading
    "num_gpus": 1,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dtype": "float16",
    "debug": False,
    # Generation parameters
    "use_chat_template": False,
    "temperature": 0,
    "repetition_penalty": 1.0,
    "max_model_len": 4096,      # Maximum input size
    "max_new_tokens": 512,      # Maximum output size
}


################################################################################
#                                   Classes                                    #
################################################################################
class LLMGeneration:
    def __init__(
            self,
            data_path,
            dataset_name,
            model_path,
            **overwrite_config,
        ):
        """
        Initialize the LLMGeneration class.

        Parameters
        ----------
        data_path : str
            Path to the dataset.
        dataset_name : str
            Name of the dataset.
        model_path : str
            Model name, or path to HuggingFace model
        **overwrite_config : Any
            Keyword arguments, which includes:
            model_provider : str
                One of local hosting: ("vllm", "huggingface", "vptq"), or one
                of online hosting: ("deepinfra", "replicate", "other")
            repetition_penalty : float, optional
                Repetition penalty, default is 1.0.
            num_gpus : int, optional
                Number of GPUs to use, default is 1.
            max_new_tokens : int, optional
                Number of max new tokens generated, default is 512.
            debug : bool, optional
                Whether to print debug messages or not. Default is False.
        """
        # Check that dataset is valid
        if not (dataset_name == "all" or dataset_name in ALL_DATASETS):
            raise RuntimeError(f"Dataset name `{dataset_name}` is invalid! Must be one of `{ALL_DATASETS}`")

        self.model_name = ""
        self.model_path = model_path        # model path. If using huggingface model, it should be the model path. Otherwise, it should be the model name.
        self.data_path = data_path          # path to the dataset
        self.dataset_name = dataset_name    # the dataset name, e.g., "winobias"

        # Store configuration
        self.llm_config = DEFAULT_CONFIG.copy()
        self.llm_config.update(**overwrite_config)

        # Get the model name according to the model path
        self.model_name = extract_model_name(self.model_path, self.llm_config["model_provider"])

        # Model related parameters to fill in
        # 1. vLLM engine
        self.vllm = None
        # 2. HuggingFace model/tokenizer (loaded by FastChat or VPTQ)
        self.hf_model = None
        self.hf_tokenizer = None

        # Load models
        self.load_model_and_tokenizer()


    def load_model_and_tokenizer(self):
        """
        Loads model and tokenizer
        """
        model_provider = self.llm_config["model_provider"]
        # CASE 1: Using vLLM instance, if needed
        if model_provider == "vllm":
            LOGGER.debug("Using VLLM model for generation. Load model from: %s", self.model_path)
            LOGGER.debug(f"Loading onto {self.llm_config['num_gpus']} GPUs...")
            self.vllm = LLM(
                model=self.model_path,
                tensor_parallel_size=self.llm_config["num_gpus"],
                dtype=self.llm_config["dtype"],
                max_model_len=self.llm_config["max_model_len"],
                guided_decoding_backend="lm-format-enforcer",
            )
            return

        # CASE 2: Loading HuggingFace model using FastChat
        if model_provider == "huggingface":
            hf_kwargs = {
                k:v for k,v in self.llm_config.items()
                if k in ["device", "num_gpus", "dtype", "max_gpu_memory", "debug"]
            }
            self.hf_model, self.hf_tokenizer = load_model(self.model_path, **hf_kwargs)
            return

        # CASE 3: Loading VPTQ-compressed HuggingFace model
        if model_provider == "vptq":
            # Late import to prevent import slowdown
            try:
                import vptq
            except ImportError:
                raise ImportError("Please install vptq: `pip install vptq`")

            # Load model and tokenizer
            self.hf_model = vptq.AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                device_map="auto",
                # torch_dtype=self.llm_config["dtype"],  # NOTE: Not supported by VPTQ library
                debug=self.llm_config["debug"],
            )
            self.hf_tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            return


    ############################################################################
    #                           HuggingFace API                                #
    ############################################################################
    def huggingface_generate(self, prompt, temperature):
        """
        Generates a response using a HuggingFace model.
        """
        model, tokenizer = self.hf_model, self.hf_tokenizer

        # Convert to chat format
        if self.llm_config["use_chat_template"]:
            prompt = llm_gen_utils.prompt2conversation(self.model_path, prompt)

        # Tokenize the input prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(self.llm_config["device"])

        # Generate output ids
        output_ids = model.generate(
            **inputs,
            do_sample=temperature > 1e-5,
            temperature=temperature or self.llm_config["temperature"],
            repetition_penalty=self.llm_config["repetition_penalty"],
            max_new_tokens=self.llm_config["max_new_tokens"],
        )

        # Adjust output ids for decoder-only models
        if not model.config.is_encoder_decoder:
            output_ids = output_ids[:, inputs["input_ids"].shape[-1]:]

        # Decode the output ids to text
        text_response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return text_response


    ############################################################################
    #                                 vLLM                                     #
    ############################################################################
    def create_vllm_kwargs(self, choices=None):
        """
        Create keyword arguments for `vllm.generate()`.

        Parameters
        ----------
        choices : list of str, optional
            If provided, the VLLM will generate a response according to
            the provided choices. The order of the choices is preserved.

        Returns
        -------
        generate_kwargs : dict or None
            Keywords arguments to pass to `vllm.generate()`. If returns None,
            then that implies that batched requests cannot be supported due
            to some unmatched keyword arguments.
        """
        generate_kwargs = {}

        # Create sampling parameters
        generate_kwargs["sampling_params"] = SamplingParams(
            temperature=self.llm_config["temperature"],
            max_tokens=self.llm_config["max_new_tokens"],
            seed=1,
        )

        # Guided Decoding
        # CASE 0: Batched processing decoding requires all share the same choices
        if choices and isinstance(choices, list) and isinstance(choices[0], list):
            final_choices = choices[0]
            for curr_choices in choices[1:]:
                if curr_choices != final_choices:
                    LOGGER.debug("Not all choices in the batch are the same! Returning None")
                    return None
            LOGGER.debug("[vLLM Generate] Flattening choices across batched requests...")
            choices = final_choices

        # CASE 1: Single-row
        if choices:
            assert isinstance(choices, list) and len(choices) > 0, \
                f"Choices must be a non-empty list! Invalid input: `{choices}`"
            assert len(choices) >= 2, "There must be 2+ choices!"
            generate_kwargs["guided_options_request"] = {
                "guided_choice": choices,
            }
        return generate_kwargs


    def vllm_generate_single(self, prompt, **kwargs):
        """
        Generates a response using a VLLM model.
        """
        # Create vLLM arguments
        gen_kwargs = self.create_vllm_kwargs(**kwargs)

        # Convert to chat format
        if self.llm_config["use_chat_template"]:
            prompt = llm_gen_utils.prompt2conversation(self.model_path, prompt)

        # Use vLLM to generate
        response = self.vllm.generate(prompt, **gen_kwargs)
        return response[0].outputs[0].text
    

    def vllm_generate_multiple(self, prompts, **kwargs):
        """
        Generates multiple responses using a VLLM model.

        Parameters
        ----------
        prompts : list of str
            The prompts to generate responses for.
        **kwargs : Any
            Additional keyword arguments for vLLM generation

        Returns
        -------
        list of str
            The generated responses.
        """
        # Create vLLM arguments
        gen_kwargs = self.create_vllm_kwargs(**kwargs)
        # CASE 1: Unable to do batched requests on texts, default to single
        if gen_kwargs is None:
            ret = []
            for idx, prompt in enumerate(prompts):
                curr_kwargs = {
                    k: (v[idx] if len(v) == len(prompts) else v)
                    for k, v in kwargs.items()
                }
                ret.append(self.vllm_generate_single(prompt, **curr_kwargs))
            return ret

        # CASE 2: Batched requests is possible
        # Convert to chat format
        if self.llm_config["use_chat_template"]:
            convert_func = llm_gen_utils.prompt2conversation
            prompts = [convert_func(self.model_path, p) for p in prompts]

        responses = self.vllm.generate(prompts, **gen_kwargs)
        return [res.outputs[0].text for res in responses]


    ############################################################################
    #                        Other Helper Functions                            #
    ############################################################################
    def generate_single(self, prompt, temperature=None, **kwargs):
        """
        Generates a response using a given model.

        Parameters
        ----------
        prompt : str
            The input text prompt for the model.
        temperature : float, optional
            The temperature setting for text generation. Default is None.

        Returns
        -------
        str
            The generated text as a string.
        """
        try:
            model_provider = self.llm_config["model_provider"]

            # Generate LLM response
            response = None
            # CASE 1: vLLM
            if model_provider == "vllm":
                # NOTE: Temperature isn't passed into vLLM
                response = self.vllm_generate_single(prompt, **kwargs)
            # CASE 2: Online Models
            if is_provider_online(model_provider):
                if kwargs.get("choices"):
                    raise NotImplementedError("Choices is not supported for online models!")
                response = llm_gen_utils.gen_online(
                    self.model_name,
                    prompt=prompt,
                    temperature=temperature,
                    replicate=(model_provider == "replicate"),
                    deepinfra=(model_provider == "deepinfra"),
                )
            # CASE 3: HuggingFace / VPTQ
            elif model_provider in ["huggingface", "vptq"]:
                # Choices is not implemented for HuggingFace generation
                if kwargs.get("choices"):
                    raise NotImplementedError(f"Choices is not supported for provider `{model_provider}`!")
                response = self.huggingface_generate(prompt, temperature)
            else:
                raise RuntimeError("This branch should never execute!")

            # Check if the response is valid before returning
            if not response:
                raise ValueError("The response is NULL or an empty string!")
            return response
        except Exception:
            tb = traceback.format_exc()
            LOGGER.debug(tb)


    def process_row(self, row, index, temperature, key_name="prompt"):
        """
        Process a single row of input data, generating a response using the
        current model if the "res" key doesn't exist or its value is empty.

        Parameters
        ----------
        row : dict
            The input data element.
        index : int
            The index of the element in the input data.
        temperature : float
            The temperature setting for text generation.
        key_name : str, optional
            The key to use for accessing the prompt in the input data element.
            Default is "prompt".
        """
        try:
            # If "res" key doesn"t exist or its value is empty, generate a new response
            if "res" not in row or not row["res"]:
                # Prepare arguments
                kwargs = {}
                # 1. Choices
                if "choices" in row:
                    kwargs["choices"] = row["choices"]

                # Perform generation
                res = self.generate_single(
                    prompt=row[key_name],
                    temperature=temperature,
                    **kwargs
                )
                if res:
                    row["res"] = res
        except Exception as e:
            # Print error message if there"s an issue during processing
            LOGGER.debug(f"Error processing element at index {index}: {e}")
            row["num_attempts"] += 1


    def process_rows(self, rows, temperature, key_name="prompt"):
        """
        Process a list of input data rows, generating responses using the
        current model if the "res" key doesn't exist or its value is empty.

        Parameters
        ----------
        rows : list
            The input data elements.
        temperature : float
            The temperature setting for text generation.
        key_name : str, optional
            The key to use for accessing the prompt in the input data elements.
            Default is "prompt".
        """
        # Filter for rows without a LLM response
        filtered_rows = [row for row in rows if not row.get("res")]

        # Early exit, if no rows to process
        if not filtered_rows:
            LOGGER.debug("No grouped rows to process")
            return

        model_provider = self.llm_config["model_provider"]
        # CASE 1: If vLLM, pass multiple row prompts at once
        if model_provider == "vllm":
            # Prepare arguments
            kwargs = {}
            # 1. Prompts
            prompts = [row["prompt"] for row in filtered_rows]
            # 2. Choices
            if "choices" in filtered_rows[0]:
                kwargs["choices"] = [row["choices"] for row in filtered_rows]

            # Perform generation
            llm_responses = self.vllm_generate_multiple(prompts, **kwargs)

            # Check responses
            for idx, row in enumerate(filtered_rows):
                # CASE 1: Valid non-empty response
                if llm_responses[idx]:
                    row["res"] = llm_responses[idx]
                # CASE 2: Empty string returned
                elif llm_responses[idx] == "":
                    row["res"] = llm_responses[idx]
                    row["num_attempts"] += 1
                # CASE 3: Returned None
                # NOTE: This branch may never execute
                else:
                    row["num_attempts"] += 1
            return

        # CASE 2: If online model, send parallel requests to LLM API
        if is_provider_online(model_provider):
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(filtered_rows)) as executor:
                futures = [
                    executor.submit(
                            self.process_row,
                            row=row,
                            index=idx,
                            temperature=temperature,
                            key_name=key_name,
                        )
                    for idx, row in enumerate(filtered_rows)
                ]

                # Wait for all futures to complete
                concurrent.futures.wait(futures)
            return

        # TODO: CASE 3: If FastChat, pass multiple prompts sequentially

        raise NotImplementedError()
        return


    ############################################################################
    #                            Main Functions                                #
    ############################################################################
    def process_file(self, input_path, output_path, file_config=None, prompt_key="prompt"):
        """
        Processes a file containing multiple data points for text generation.

        Args:
            input_path (str): Path to the input data file.
            output_path (str): Path where the processed data will be saved.
            file_config (dict): Configuration settings for file processing.
            prompt_key (str): The key in the dictionary where the prompt is located.
        """
        # INPUT: Sanitize file_config
        file_config = file_config or {}
        temperature = file_config.get(os.path.basename(input_path), 0.0)

        # Check if the input file exists
        if not os.path.exists(input_path):
            raise RuntimeError(f"File {input_path} does not exist")

        # Check if the output file already exists
        if os.path.exists(output_path):
            LOGGER.info(f"Resuming from {output_path}")
            saved_data = json_utils.load_json(output_path)
        else:
            # Load the input data from the file
            LOGGER.info(f"Loading data from {input_path}")
            saved_data = json_utils.load_json(input_path)

        # Early return, if all rows are done
        if all([bool(row.get("res")) for row in saved_data]):
            LOGGER.info("Already done, returning early!")
            return

        # Create thread lock
        lock = threading.Lock()

        # Process the input data in groups
        batch_size = 32 if (self.llm_config["model_provider"] == "vllm") else NUM_WORKERS

        # Add number of attempts to each row
        MAX_ATTEMPTS = 3
        for row in saved_data:
            row["num_attempts"] = 0

        while not all(bool(row.get("res")) for row in saved_data):
            curr_idx = 0
            grouped_rows = []
            while curr_idx < len(saved_data) and len(grouped_rows) < batch_size:
                row = saved_data[curr_idx]
                # Include if no valid response yet and hasn't failed excessively
                if row.get("num_attempts") < MAX_ATTEMPTS and \
                        not row.get("res") or row.get("res") == "null":
                    grouped_rows.append(row)
                curr_idx += 1

            # If no more data, then break before creating threads
            if not grouped_rows:
                break

            # Process the grouped rows
            self.process_rows(grouped_rows, temperature, prompt_key)

            # Save the updated saved data to the output file
            json_utils.save_json(saved_data, output_path, lock=lock)
            LOGGER.debug(f"Processed {input_path} and saved results to {output_path}")

        # Save the updated saved data to the output file
        json_utils.save_json(saved_data, output_path, lock=lock)
        LOGGER.debug(f"Processed {input_path} and saved results to {output_path}")


    def infer_dataset_single(self, dataset_name):
        """
        Executes a single test based on specified parameters.

        Note
        ----
        Each dataset can have multiple sub-folders for each sub-group
        (e.g., age.json, gender.json)

        Parameters
        ----------
        dataset_name : str
            The name of the dataset to use for evaluation.
        """
        LOGGER.info(f"Beginning generation with `{dataset_name}` evaluation at temperature {self.llm_config['temperature']}.")
        LOGGER.info(f"Evaluating target model: {self.model_name}")

        base_dir = os.path.join(self.data_path, dataset_name)
        result_dir = os.path.join("generation_results", self.model_name, dataset_name)
        os.makedirs(result_dir, exist_ok=True)

        json_paths = list(set(glob.glob(os.path.join(base_dir, "*.json"))))
        for json_path in tqdm(json_paths):
            LOGGER.info("Processing file: %s", json_path)
            save_path = os.path.join(result_dir, os.path.basename(json_path))
            self.process_file(json_path, save_path)


    def infer_dataset(self, max_retries=2, retry_interval=3):
        """
        Run the generation task for the specified dataset(s).

        Parameters
        ----------
        max_retries : int
            The maximum number of times to retry if the test fails.
        retry_interval : int
            The time in seconds to wait before retrying.
        """
        if not os.path.exists(self.data_path):
            LOGGER.debug(f"Dataset path {self.data_path} does not exist.")
            return None

        # If all datasets specified, then run generate for all datasets
        datasets = ALL_DATASETS if self.dataset_name == "all" else [self.dataset_name]

        # Iterate for each dataset
        for dataset in datasets:
            num_retries = 0
            successful = False
            while not successful and num_retries < max_retries:
                try:
                    self.infer_dataset_single(dataset)
                    successful = True
                except Exception as e:
                    LOGGER.error(f"Test function failed on attempt {num_retries + 1}: {e}")
                    LOGGER.error(f"Retrying in {retry_interval} seconds...")
                    time.sleep(retry_interval)
                    num_retries += 1
            if not successful:
                LOGGER.error("Test failed after maximum retries.")


################################################################################
#                               Helper Functions                               #
################################################################################
def is_provider_online(model_provider):
    """
    Check if the model provider is an online provider.

    Parameters
    ----------
    model_provider : str
        Name of model provider

    Returns
    -------
    bool
        True if the model provider is one of the online providers
    """
    return model_provider in ["deepinfra", "replicate", "other"]


def extract_model_name(model_path, model_provider="vllm"):
    """
    Extract model name from model path.

    Parameters
    ----------
    model_path : str
        Path to the model
    model_provider : str
        Model provider name

    Returns
    -------
    str
        Model name

    Raises
    ------
    RuntimeError
        If the model path is not found in the model mapping
    """
    # Get the model name according to the model path
    model_name = None
    model_mapping = MODEL_INFO["model_mapping"]
    if model_path in model_mapping:
        model_name = model_mapping[model_path]
    elif model_path.split("/")[-1] in model_mapping:
        model_name = model_mapping[model_path.split("/")[-1]]
    else:
        raise RuntimeError(
            "Please ensure model path has mapping in `src/config/config.py`!"
            f"\n\tModel Path: `{model_path}`")

    # Ensure model name is valid, if online model is chosen
    if is_provider_online(model_provider):
        assert model_name in MODEL_INFO['online_model'], (
            f"Online model provided `{model_name}` is invalid! "
            f"\nValid options: {MODEL_INFO['online_model']}"
        )

    return model_name
