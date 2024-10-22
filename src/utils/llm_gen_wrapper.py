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
from tqdm import tqdm
from vllm import LLM, SamplingParams

# Custom libraries
from src.config import config
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

# General test types
TEST_TYPES = ["Continuation", "Conversation", "Recognition", "Selection"]
# Names of Stereotype Datasets
STEREOTYPE_DATASETS = [f"CEB-{test}-S" for test in TEST_TYPES] + [
    "CEB-Adult",
    "CEB-Credit",
    # TODO: Handle later
    # "CEB-RB-Recognition",
    # "CEB-WB-Recognition",
    # "CEB-CP-Recognition",
]
# Names of Toxicity Datasets
TOXICITY_DATASETS = [f"CEB-{test}-T" for test in TEST_TYPES] + [
    "CEB-Jigsaw",
    # TODO: Handle later
    # "CEB-SS-Recognition",
]
# Names of all datasets
ALL_DATASETS = STEREOTYPE_DATASETS + TOXICITY_DATASETS

# Data type to load LLM
DTYPE = "float16"          # "bfloat16", "float16"

# Number of threads to use in sending requests to LLM APIs
NUM_WORKERS = 8

################################################################################
#                                   Classes                                    #
################################################################################
class LLMGeneration:
    def __init__(
            self,
            data_path,
            dataset_name,
            model_path,
            online_model=False,
            use_deepinfra=False,
            use_replicate=False,
            use_vllm=False,
            repetition_penalty=1.0,
            num_gpus=1,
            max_new_tokens=512,
            debug=False,
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
            Path to the model. If using huggingface model, it should be the model path. Otherwise, it should be the model name.
        online_model : bool, optional
            Whether to use the online model or not. Default is False.
        use_deepinfra : bool, optional
            Whether to use the deepinfra API or not. Default is False.
        use_replicate : bool, optional
            Whether to use the replicate API or not. Default is False.
        use_vllm : bool, optional
            Whether to use the vLLM to run huggingface models or not. Default is False.
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
        self.online_model = online_model
        self.temperature = 0                                            # temperature setting for text generation, default 0.0 (greedy decoding)
        self.repetition_penalty = repetition_penalty                    # repetition penalty, default is 1.0
        self.num_gpus = num_gpus
        self.max_new_tokens = max_new_tokens                            # Number of max new tokens generated
        self.debug = debug
        self.online_model_list = llm_gen_utils.get_models()[1]                        # Online model list, typically contains models that are not huggingface models
        self.model_mapping = llm_gen_utils.get_models()[0]                            # Mapping between model path and model name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_replicate = use_replicate                              # Temporarily set to False as we don"t use replicate api
        self.use_deepinfra = use_deepinfra                              # Temporarily set to False as we don"t use deepinfra api
        self.use_vllm = use_vllm                                        # Set this to be True when using vLLM to run huggingface models

        # Get the model name according to the model path
        self.model_name = None
        model_mapping = config.MODEL_INFO["model_mapping"]
        if self.model_path in model_mapping:
            self.model_name = model_mapping[self.model_path]
        elif self.model_path.split("/")[-1] in model_mapping:
            self.model_name = model_mapping[self.model_path.split("/")[-1]]
        else:
            raise RuntimeError(
                "Please ensure model path has mapping in `src/config/config.py`!"
                f"\n\tModel Path: `{self.model_path}`")

        # Model related parameters to fill in
        self.vllm = None
        self.sampling_params = None
        self._model = None
        self._tokenizer = None

        # Load models
        self.load_model_and_tokenizer()


    def load_model_and_tokenizer(self):
        """
        Loads model and tokenizer
        """
        # CASE 1: Instantiate vLLM instance, if needed
        if self.use_vllm:
            LOGGER.debug("Using VLLM model for generation. Load model from: %s", self.model_path)
            LOGGER.debug(f"Loading onto {self.num_gpus} GPUs...")
            self.vllm = LLM(
                model=self.model_path,
                tensor_parallel_size=self.num_gpus,
                dtype=DTYPE,
            )
            self.sampling_params = SamplingParams(temperature=self.temperature,
                                                  top_p=1.0, seed=1,
                                                  max_tokens=self.max_new_tokens)

        # Early return, if model is already loaded
        if self._model is not None or self._tokenizer is not None:
            return None

        # CASE 1: Using online models without using replicate or deepinfra apis
        if self.online_model:
            assert self.model_name in self.online_model_list, \
                f"Online model provided `{self.model_name}` is not in " \
                f"online model list! \n\t{self.online_model_list}"
            return
        # CASE 2: Using vLLM
        elif self.use_vllm:
            return

        # CASE 3: Loading using FastChat
        model, tokenizer = load_model(
            self.model_path,
            self.online_model,
            self.use_replicate,
            self.use_deepinfra,
            self.use_vllm,
            self.num_gpus,
            self.device,
            self.debug,
        )
        self._model = model
        self._tokenizer = tokenizer


    def _generation_hf(self, prompt, temperature):
        """
        Generates a response using a HuggingFace model.
        """
        model, tokenizer = self._model, self._tokenizer

        prompt = llm_gen_utils.prompt2conversation(self.model_path, prompt)
        inputs = tokenizer([prompt])
        inputs = {k: torch.tensor(v).to(self.device) for k, v in inputs.items()}
        output_ids = model.generate(
            **inputs,
            do_sample=True if temperature > 1e-5 else False,
            temperature=temperature,
            repetition_penalty=self.repetition_penalty,
            max_new_tokens=self.max_new_tokens,
        )
        if model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(inputs["input_ids"][0]):]
        outputs = tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )
        return outputs


    def _generation_vllm(self, prompt):
        """
        Generates a response using a VLLM model.
        """
        response = self.vllm.generate(prompt, self.sampling_params)
        return response[0].outputs[0].text
    

    def _generation_vllm_multiple(self, prompts):
        """
        Generates multiple responses using a VLLM model.

        Parameters
        ----------
        prompts : list of str
            The prompts to generate responses for.

        Returns
        -------
        list of str
            The generated responses.
        """
        responses = self.vllm.generate(prompts, self.sampling_params)
        return [res.outputs[0].text for res in responses]


    def generate_single(self, prompt, temperature=None):
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
            # CASE 1: Online Models
            if self.online_model:
                ans = llm_gen_utils.gen_online(self.model_name,
                                 prompt, temperature,
                                 replicate=self.use_replicate,
                                 deepinfra=self.use_deepinfra)
            # CASE 2: vLLM
            elif self.use_vllm:
                ans = self._generation_vllm(prompt)
            # CASE 3: HuggingFace
            else:
                ans = self._generation_hf(prompt, temperature)
            if not ans:
                raise ValueError("The response is NULL or an empty string!")
            return ans
        except Exception as e:
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
                res = self.generate_single(prompt=row[key_name], temperature=temperature)
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

        # CASE 1: If vLLM, pass multiple row prompts at once
        if self.use_vllm:
            prompts = [row["prompt"] for row in filtered_rows]
            llm_responses = self._generation_vllm_multiple(prompts)
            for idx, row in enumerate(filtered_rows):
                if llm_responses[idx]:
                    row["res"] = llm_responses[idx]
                else:
                    row["num_attempts"] += 1
            return

        # CASE 2: If online model, send parallel requests to LLM API
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
        batch_size = 32 if self.use_vllm else NUM_WORKERS

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
        LOGGER.info(f"Beginning generation with `{dataset_name}` evaluation at temperature {self.temperature}.")
        LOGGER.info(f"Evaluating target model: {self.model_name}")

        base_dir = os.path.join(self.data_path, dataset_name)
        section = os.path.basename(base_dir)
        result_dir = os.path.join("generation_results", self.model_name, section)
        os.makedirs(result_dir, exist_ok=True)

        file_list = glob.glob(os.path.join(base_dir, "*.json"))
        for file_path in tqdm(file_list, desc="Processing files"):
            LOGGER.info("Processing file: %s", file_path)
            file_name = os.path.basename(file_path)
            save_path = os.path.join(result_dir, file_name)
            self.process_file(file_path, save_path)


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
                    LOGGER.debug(f"Test function failed on attempt {num_retries + 1}: {e}")
                    LOGGER.debug(f"Retrying in {retry_interval} seconds...")
                    time.sleep(retry_interval)
            if not successful:
                LOGGER.debug("Test failed after maximum retries.")
