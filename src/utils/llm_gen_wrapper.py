# Standard libraries
import concurrent
import glob
import multiprocessing
import logging
import os
import threading
import time

# Non-standard libraries
import numpy as np
import torch
import traceback
import urllib3
from dotenv import load_dotenv
from fastchat.model import load_model
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Custom libraries
from config import (
    MODEL_INFO, ALL_CEB_DATASETS, OPEN_ENDED_DATASETS, CLOSE_ENDED_DATASETS, ALL_FMT_DATASETS, ALL_DE_DATASETS,
    DIR_GENERATIONS, DIR_MODELS,
    SYSTEM_PROMPT_MAP, FMT_USER_KEY, FMT_ASSISTANT_KEY,
)
from src.utils import json_utils, llm_gen_utils


# Disable warnings
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

# Default configuration parameters
DEFAULT_CONFIG = {
    "model_provider": "vllm",
    # Local model loading
    "num_gpus": 1,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dtype": "bfloat16",
    "max_num_seqs": 16,         # Maximum number of concurrent requests
    "gpu_memory_utilization": 0.95,
    "enable_chunked_prefill": True,
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
            model_path_or_name,
            system_prompt_type="no_sys_prompt",
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
        model_path_or_name : str
            Model name, or path to HuggingFace model
        system_prompt_type : str, optional
            Type of system prompt to use
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
        dataset_names = []
        self.dataset_collection = "ceb"
        if dataset_name == "all_ceb":
            dataset_names.extend(ALL_CEB_DATASETS)
        elif dataset_name == "all_ceb_open_ended":
            dataset_names.extend(OPEN_ENDED_DATASETS)
        elif dataset_name == "all_ceb_close_ended":
            dataset_names.extend(CLOSE_ENDED_DATASETS)
        elif dataset_name == "all_fmt":
            self.dataset_collection = "fmt"
            # Overwrite parameters to faithfully follow FMT-Bench paper
            LOGGER.info("Overwriting parameters to faithfully follow FMT-Bench paper (temperature=0.7, top_k=1)")
            overwrite_config["max_new_tokens"] = 150
            # NOTE: This is functionally equivalent to temperature = 0, but is kept for consistency
            overwrite_config["temperature"] = 0.7
            overwrite_config["top_k"] = 1
            dataset_names.extend(ALL_FMT_DATASETS)
        elif dataset_name == "de":
            self.dataset_collection = "de"
            dataset_names = ALL_DE_DATASETS
            LOGGER.info("Overwriting parameters for DiscrimEval close-ended datasets (temperature=1, chat template=False)")
            overwrite_config["temperature"] = 1
            overwrite_config["use_chat_template"] = False
        else:
            assert dataset_name in ALL_CEB_DATASETS, f"Dataset name `{dataset_name}` is invalid! Must be one of `{ALL_CEB_DATASETS}`"
            dataset_names.append(dataset_name)

        # Path to dataset
        self.data_path = data_path
        # Name of dataset/s
        self.dataset_names = dataset_names

        # Type of system prompt
        self.system_prompt_type = system_prompt_type
        # System prompt for dataset (to fill)
        # NOTE: Only for chat datasets
        self.system_prompt = None

        # Store configuration
        self.llm_config = DEFAULT_CONFIG.copy()
        self.llm_config.update(**overwrite_config)

        # Get the model name according to the model path, or vice versa
        # NOTE: If using huggingface model, it should be the model path. Otherwise, it should be the model name.
        self.model_name, self.model_path = extract_model_path_or_name(
            model_path_or_name,
            self.llm_config["model_provider"],
            self.llm_config["use_chat_template"]
        )

        # If local model, check if it's stored in the `models` directory
        # NOTE: If it is, then update the model path
        if self.llm_config["model_provider"] in ["vllm", "huggingface", "vptq"]:
            if os.path.exists(self.model_path):
                pass
            elif os.path.exists(os.path.join(DIR_MODELS, self.model_path)):
                self.model_path = os.path.join(DIR_MODELS, self.model_path)

        # Model related parameters to fill in
        # NOTE: Lazy loaded for efficiency
        self.is_model_loaded = False
        # 1. vLLM engine
        self.vllm = None
        # 2. HuggingFace model/tokenizer (loaded by FastChat or VPTQ)
        self.hf_model = None
        self.hf_tokenizer = None


    def ensure_model_loaded(self):
        """
        Loads model (and tokenizer), if not already
        """
        # Simply return, if model is already loaded
        if self.is_model_loaded:
            return

        model_provider = self.llm_config["model_provider"]
        # CASE 1: Using vLLM instance, if needed
        if model_provider == "vllm":
            LOGGER.debug("Using VLLM model for generation. Load model from: %s", self.model_path)
            LOGGER.debug(f"Loading onto {self.llm_config['num_gpus']} GPUs...")

            # Specify additional keyword arguments, if model is a mistral model
            additional_llm_kwargs = {}
            # HACK: Currently, MistralTokenizer does not support `apply_chat_template`
            #       so we'll rely on the `vLLM/HuggingFace` package instead of `mistral-common`
            # if "mistral" in self.model_path.lower():
            #     additional_llm_kwargs["tokenizer_mode"] = "mistral"
            #     additional_llm_kwargs["config_format"] = "mistral"
            #     additional_llm_kwargs["load_format"] = "mistral"

            # If multiple GPUs, enforce eager to save as much memory as possible
            if self.llm_config["num_gpus"] > 1:
                additional_llm_kwargs["enforce_eager"] = True

            try:
                self.vllm = LLM(
                    model=self.model_path,
                    tensor_parallel_size=self.llm_config["num_gpus"],
                    dtype=self.llm_config["dtype"],
                    max_model_len=self.llm_config["max_model_len"],
                    gpu_memory_utilization=self.llm_config["gpu_memory_utilization"],
                    max_num_seqs=self.llm_config["max_num_seqs"],
                    enable_chunked_prefill=self.llm_config["enable_chunked_prefill"],
                    trust_remote_code=True,
                    **additional_llm_kwargs,
                )
            except Exception:
                LOGGER.critical("[LLMGeneration] Failed to load vLLM model! Exiting early...")
                tb = traceback.format_exc()
                LOGGER.error(tb)
                exit(1)
            self.is_model_loaded = True
            return

        # CASE 2: Loading HuggingFace model using FastChat
        if model_provider == "huggingface":
            hf_kwargs = {
                k:v for k,v in self.llm_config.items()
                if k in ["device", "num_gpus", "dtype", "max_gpu_memory", "debug"]
            }
            self.hf_model, self.hf_tokenizer = load_model(self.model_path, **hf_kwargs)
            self.is_model_loaded = True
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
            self.is_model_loaded = True
            return

        raise NotImplementedError(f"[LLM Generation Wrapper] Model provider {model_provider} not currently supported!")


    ############################################################################
    #                           HuggingFace API                                #
    ############################################################################
    def huggingface_generate(self, prompt, temperature):
        """
        Generates a response using a HuggingFace model.
        """
        self.ensure_model_loaded()
        model, tokenizer = self.hf_model, self.hf_tokenizer

        # Convert to chat format
        if self.llm_config["use_chat_template"]:
            prompt = apply_chat_template_hf(self.model_path, prompt)

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
    def create_vllm_kwargs_choice(self, **extra_sampling_kwargs):
        """
        Create keyword arguments for when using `vllm.generate()` for computing
        log probabilities.

        Parameters
        ----------
        **extra_sampling_kwargs : Any
            Keyword arguments to override in SamplingParams

        Returns
        -------
        generate_kwargs : dict or None
            Keywords arguments to pass to `vllm.generate()`. If returns None,
            then that implies that batched requests cannot be supported due
            to some unmatched keyword arguments.
        """
        generate_kwargs = {}

        LOGGER.info(["vLLM Generate] Choices are provided! Using prompt_logprobs to get choice probability..."])
        # Create sampling parameters
        sampling_kwargs = {
            "temperature": 1,
            "prompt_logprobs": 1,
            "max_tokens": 1,
            "seed": 1
        }
        # Overwrite sampling parameters
        sampling_kwargs.update(extra_sampling_kwargs)
        generate_kwargs["sampling_params"] = SamplingParams(**sampling_kwargs)

        return generate_kwargs


    def create_vllm_kwargs_generate(self, **extra_sampling_kwargs):
        """
        Create keyword arguments for `vllm.generate()` or `vllm.chat()`.

        Parameters
        ----------
        **extra_sampling_kwargs : Any
            Keyword arguments to override in SamplingParams

        Returns
        -------
        generate_kwargs : dict or None
            Keywords arguments to pass to `vllm.generate()`. If returns None,
            then that implies that batched requests cannot be supported due
            to some unmatched keyword arguments.
        """
        generate_kwargs = {}

        # Create sampling parameters
        sampling_kwargs = {
            "temperature": self.llm_config["temperature"],
            "max_tokens": self.llm_config["max_new_tokens"],
            "seed": 1
        }
        # Overwrite sampling parameters
        sampling_kwargs.update(extra_sampling_kwargs)
        generate_kwargs["sampling_params"] = SamplingParams(**sampling_kwargs)

        return generate_kwargs


    def vllm_logprob_single(self, prompt, choice, **kwargs):
        """
        Given a promt and choice of continuation, compute log probability of
        provided continuation.

        Parameters
        ----------
        prompt : str
            Initial prompt string
        choice : str
            Choice of continuation directly from the prompt
        """
        self.ensure_model_loaded()

        # Create vLLM arguments
        gen_kwargs = self.create_vllm_kwargs_choice(**kwargs)

        # Convert to chat format
        if self.llm_config["use_chat_template"]:
            prompt = apply_chat_template_vllm(self.vllm, prompt, self.system_prompt)
        # If not chat format, prepend system prompt (if available)
        elif self.system_prompt:
            prompt = self.system_prompt + "\n\n" + prompt

        # Append choice text
        # NOTE: Add space in between if no space already at the end of the prompt
        between_text = "" if prompt[-1].isspace() else " "
        prompt = prompt + between_text
        final_prompt = prompt + choice

        # Use vLLM to generate
        response = self.vllm.generate(final_prompt, **gen_kwargs)

        # Prepare return
        ret = {
            "res": choice,
        }

        # Get average probability for the choice
        prompt_logprobs = response[0].prompt_logprobs
        ret["res_seq_prob"] = extract_choice_logprobs(choice, prompt_logprobs)
        return ret


    def vllm_logprob_multiple(self, prompts, choice, **kwargs):
        """
        Computes log probability for the same set choices between multiple prompts.

        Parameters
        ----------
        prompts : list of str
            The prompts to generate responses for.
        choice : str
            Same hypothetical text continuation across multiple prompts
        **kwargs : Any
            Additional keyword arguments for vLLM generation

        Returns
        -------
        list of dict
            List of generated responses.
        """
        assert isinstance(choice, str), "Only one choice text is possible for `vllm_logprob_multiple()`!"
        self.ensure_model_loaded()

        # Create vLLM arguments
        gen_kwargs = self.create_vllm_kwargs_choice(**kwargs)

        # Convert to chat format
        if self.llm_config["use_chat_template"]:
            prompts = [apply_chat_template_vllm(self.vllm, p, self.system_prompt) for p in prompts]
        # If not chat format, prepend system prompt (if available)
        elif self.system_prompt:
            prompts = [self.system_prompt + "\n\n" + p for p in prompts]

        # Append choice text
        accum_final_prompts = []
        for prompt in prompts:
            # NOTE: Add space in between if no space already at the end of the prompt
            between_text = "" if prompt[-1].isspace() else " "
            prompt = prompt + between_text
            final_prompt = prompt + choice
            accum_final_prompts.append(final_prompt)

        # Generate in batch
        responses = self.vllm.generate(accum_final_prompts, **gen_kwargs)

        # Extract results
        accum_ret = []
        for curr_response in responses:
            curr_ret = {
                "res": choice,
            }
            # Get average probability for the choice
            prompt_logprobs = curr_response.prompt_logprobs
            curr_ret["res_seq_prob"] = extract_choice_logprobs(choice, prompt_logprobs)
            accum_ret.append(curr_ret)

        return accum_ret


    def vllm_generate_single(self, prompt, choices=None, **kwargs):
        """
        Generates a response using a VLLM model.
        """
        self.ensure_model_loaded()

        # Create vLLM arguments
        gen_kwargs = self.create_vllm_kwargs_generate(choices=choices, **kwargs)

        # Convert to chat format
        if self.llm_config["use_chat_template"]:
            prompt = apply_chat_template_vllm(self.vllm, prompt, self.system_prompt)
        # If not chat format, prepend system prompt (if available)
        elif self.system_prompt:
            prompt = self.system_prompt + "\n\n" + prompt

        # Use vLLM to generate
        response = self.vllm.generate(prompt, **gen_kwargs)

        # Prepare return
        ret = {
            "res": response[0].outputs[0].text,
        }

        return ret


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
        list of dict
            List of generated responses.
        """
        self.ensure_model_loaded()

        # Create vLLM arguments
        gen_kwargs = self.create_vllm_kwargs_generate(**kwargs)

        # Convert to chat format
        if self.llm_config["use_chat_template"]:
            prompts = [apply_chat_template_vllm(self.vllm, p, self.system_prompt) for p in prompts]
        # If not chat format, prepend system prompt (if available)
        elif self.system_prompt:
            prompts = [self.system_prompt + "\n\n" + p for p in prompts]

        # Generate in batch
        responses = self.vllm.generate(prompts, **gen_kwargs)

        # Extract results
        accum_ret = []
        for curr_response in responses:
            curr_ret = {
                "res": curr_response.outputs[0].text
            }
            accum_ret.append(curr_ret)
        return accum_ret


    def vllm_chat_single(self, conversation, **kwargs):
        """
        Generates a response using a VLLM model.

        Parameters
        ----------
        conversation : list of dict
            Single conversation

        Returns
        -------
        dict
            Contains generated responses.
        """
        self.ensure_model_loaded()

        # Create vLLM arguments
        gen_kwargs = self.create_vllm_kwargs_generate(**kwargs)

        # Use vLLM to chat
        response = self.vllm.chat(conversation, **gen_kwargs)

        # Prepare return
        ret = {
            "res": response[0].outputs[0].text,
        }
        return ret


    def vllm_chat_multiple(self, conversations, **kwargs):
        """
        Generates multiple responses using a VLLM model.

        Parameters
        ----------
        conversations : list of list of dict
            Multiple conversations
        **kwargs : Any
            Additional keyword arguments for vLLM generation

        Returns
        -------
        list of dict
            List of generated responses.
        """
        self.ensure_model_loaded()

        # Create vLLM arguments
        gen_kwargs = self.create_vllm_kwargs_generate(**kwargs)

        # Generate in batch
        responses = self.vllm.chat(conversations, **gen_kwargs)

        # Extract results
        accum_ret = []
        for curr_response in responses:
            curr_ret = {
                "res": curr_response.outputs[0].text,
            }
            accum_ret.append(curr_ret)
        return accum_ret


    ############################################################################
    #                        Other Helper Functions                            #
    ############################################################################
    def logprobs_single(self, prompt, choice, **kwargs):
        """
        Generates a response using a given model.

        Parameters
        ----------
        prompt : str
            The input text prompt for the model.
        choice : str
            The hypothetical continuation of the prompt

        Returns
        -------
        dict
            Contains generated text in `res` and `res_seq_prob` for choice
            choices provided for vLLM
        """
        try:
            model_provider = self.llm_config["model_provider"]

            # Get sequence probability for LLM response
            # CASE 1: vLLM
            if model_provider == "vllm":
                ret = self.vllm_logprob_single(prompt, choice, **kwargs)
            else:
                raise NotImplementedError(f"[logprobs_single] Model provider given `{model_provider}` is not yet supported!")

            # Check if the response is valid before returning
            if not ret["res"]:
                raise ValueError("The response is NULL or an empty string!")
            return ret
        except Exception:
            tb = traceback.format_exc()
            LOGGER.error(tb)


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
        dict
            Contains generated text in `res`
        """
        try:
            model_provider = self.llm_config["model_provider"]

            # Generate LLM response
            ret = {
                "res": None
            }
            # CASE 1: vLLM
            if model_provider == "vllm":
                ret = self.vllm_generate_single(prompt, temperature=temperature, **kwargs)
            # CASE 2: Online Models
            elif is_provider_online(model_provider):
                ret["res"] = llm_gen_utils.gen_online(
                    self.model_name,
                    prompt=prompt,
                    temperature=temperature,
                    replicate=(model_provider == "replicate"),
                    deepinfra=(model_provider == "deepinfra"),
                )
            # CASE 3: HuggingFace / VPTQ
            elif model_provider in ["huggingface", "vptq"]:
                ret["res"] = self.huggingface_generate(prompt, temperature)
            else:
                raise RuntimeError(f"[generate_single] Invalid model provider given! `{model_provider}`")

            # Check if the response is valid before returning
            if not ret["res"]:
                raise ValueError("The response is NULL or an empty string!")
            return ret
        except Exception:
            tb = traceback.format_exc()
            LOGGER.error(tb)


    def chat_single(self, conversation, temperature=None, **kwargs):
        """
        Generates a response to a chat using a given model.

        Parameters
        ----------
        conversation : list of dict
            Single conversation to continue by the model
        temperature : float, optional
            The temperature setting for text generation. Default is None.

        Returns
        -------
        dict
            Contains generated text in `res`
        """
        try:
            model_provider = self.llm_config["model_provider"]

            # Generate LLM response
            ret = {
                "res": None
            }
            # CASE 1: vLLM
            if model_provider == "vllm":
                ret = self.vllm_chat_single(conversation, temperature=temperature, **kwargs)
            # CASE 2: Online Models
            elif is_provider_online(model_provider):
                raise NotImplementedError("Online models with chat format is not yet supported!")
                # ret["res"] = llm_gen_utils.gen_online(
                #     self.model_name,
                #     conversation=conversation,
                #     temperature=temperature,
                #     replicate=(model_provider == "replicate"),
                #     deepinfra=(model_provider == "deepinfra"),
                # )
            # CASE 3: HuggingFace / VPTQ
            elif model_provider in ["huggingface", "vptq"]:
                raise NotImplementedError(f"`{model_provider}` models with chat format is not yet supported!")
                # ret["res"] = self.huggingface_generate(conversation, temperature)
            else:
                raise RuntimeError(f"[chat_single] Invalid model provider given! `{model_provider}`")

            # Check if the response is valid before returning
            if not ret["res"]:
                raise ValueError("The response is NULL or an empty string!")
            return ret
        except Exception:
            tb = traceback.format_exc()
            LOGGER.error(tb)


    ############################################################################
    #                            CEB Functions                                 #
    ############################################################################
    def process_row_ceb(self, row, index, temperature, key_name="prompt"):
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
            # If "res" key doesn't exist or its value is empty, generate a new response
            if row.get("num_attempts", 0) != 3 and row.get("res"):
                return

            # CASE 1: No choices provided
            prompt = row[key_name]
            if not row.get("choices"):
                curr_response = self.generate_single(
                    prompt=row[key_name],
                    temperature=temperature,
                )
                row["res"] = curr_response["res"]
                return

            # CASE 2: Choices are provided, send a request for every choice
            # NOTE: Store probability for each choice
            choices = row["choices"]
            choices_probs = []
            for choice in choices:
                curr_response = self.logprobs_single(prompt, choice)
                choices_probs.append(curr_response["res_seq_prob"])

            # Normalize probabilities and store choice with highest probability
            normalized_probs = [prob / sum(choices_probs) for prob in choices_probs]
            row["res"] = choices[normalized_probs.index(max(normalized_probs))]
            row["res_probs"] = [round(prob, 4) for prob in normalized_probs]
        except Exception:
            # Print error message if there"s an issue during processing
            LOGGER.error(f"[process_row_ceb] Exception occured!")
            tb = traceback.format_exc()
            LOGGER.error(tb)
            row["num_attempts"] += 1


    def process_rows_ceb(self, rows, temperature, key_name="prompt"):
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
            # Flatten choices if provided multiple rows
            choices = None
            revert_to_single = False
            if "choices" in filtered_rows[0]:
                for row in filtered_rows:
                    if choices is None:
                        choices = row["choices"]
                    elif row["choices"] != choices:
                        revert_to_single = True
                        break

            # CASE 0: If choices differ per row, revert to unbatched processing
            if revert_to_single:
                for idx, row in enumerate(filtered_rows):
                    self.process_row_ceb(row, index=idx, temperature=temperature, key_name=key_name)
                return

            # Prepare prompts
            prompts = [row["prompt"] for row in filtered_rows]

            # CASE 1: No choices provided
            if choices is None:
                llm_responses = self.vllm_generate_multiple(prompts)

                # Check responses
                for idx, row in enumerate(filtered_rows):
                    # CASE 1: Valid non-empty response
                    curr_response_text = llm_responses[idx]["res"]
                    if curr_response_text:
                        row["res"] = curr_response_text
                    # CASE 2: Empty string returned
                    elif curr_response_text == "":
                        row["res"] = curr_response_text
                        row["num_attempts"] += 1
                    else:
                        raise RuntimeError("Unexpected response from vLLM!")
                return

            # CASE 2: Choices are provided (and the same for every row), send a request for each choice
            # NOTE: Store probability for each choice
            idx_to_probs = {}
            for choice in choices:
                llm_responses = self.vllm_logprob_multiple(prompts, choice)

                # Store response for every choice
                for idx, curr_response in enumerate(llm_responses):
                    if idx not in idx_to_probs:
                        idx_to_probs[idx] = []
                    idx_to_probs[idx].append(curr_response["res_seq_prob"])

            # Now assign choice response based on sequence probabilities
            for idx, row in enumerate(filtered_rows):
                choices_probs = idx_to_probs[idx]

                # Normalize probabilities and store choice with highest probability
                normalized_probs = [prob / sum(choices_probs) for prob in choices_probs]
                row["res"] = choices[normalized_probs.index(max(normalized_probs))]
                row["res_probs"] = [round(prob, 4) for prob in normalized_probs]
            return

        # CASE 2: If online model, send parallel requests to LLM API
        if is_provider_online(model_provider):
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(filtered_rows)) as executor:
                futures = [
                    executor.submit(
                            self.process_row_ceb,
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


    def process_file_ceb(self, input_path, output_path, file_config=None, prompt_key="prompt"):
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
        temperature = file_config.get(os.path.basename(input_path), self.llm_config["temperature"])

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

        # If there are options, ensure they're only two choices at most
        for row in saved_data:
            if "choices" in row:
                assert len(row["choices"]) == 2, (
                    f"Found row with more than two choices in `{input_path}`\n"
                    f"Choices: {row['choices']}"
                )

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

        # Perform inference in batches
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
            self.process_rows_ceb(grouped_rows, temperature, prompt_key)

            # Save the updated saved data to the output file
            json_utils.save_json(saved_data, output_path, lock=lock)
            LOGGER.debug(f"Processed {input_path} and saved results to {output_path}")

        # Save the updated saved data to the output file
        json_utils.save_json(saved_data, output_path, lock=lock)
        LOGGER.debug(f"Processed {input_path} and saved results to {output_path}")


    ############################################################################
    #                        FairMT-Bench Functions                            #
    ############################################################################
    def process_row_fmt(self, row, index, temperature):
        """
        Move 1 conversation along by one step (e.g., 1->2 or 3->4)

        Note
        ----
        For a 5-turn conversation, you need to pass the row into this function
        5 times.

        Parameters
        ----------
        row : dict
            The input data element.
        index : int
            The index of the element in the input data.
        temperature : float
            The temperature setting for text generation.
        """
        try:
            # If "res" key doesn't exist or its value is empty, generate a new response
            if row.get("num_attempts", 0) != 3 and not is_conversation_done(row):
                return

            # Raise error, if choices provided
            if row.get("choices"):
                raise RuntimeError("Chat template is not compatible with choices currently!")

            # Get conversation till this point
            curr_turn, conversation = extract_conversation_till_turn(row)

            # Early exit, if conversation is already done
            if curr_turn is None:
                LOGGER.debug("Conversation is already ended!")
                return

            # Prepend system message
            if self.system_prompt:
                conversation.insert(0, {"role": "system", "content": self.system_prompt})

            # Generate chat response
            curr_response = self.chat_single(
                conversation=conversation,
                temperature=temperature,
            )
            response_key = FMT_ASSISTANT_KEY.format(turn=curr_turn)
            row[response_key] = curr_response["res"]
            return
        except Exception as e:
            # Print error message if there"s an issue during processing
            LOGGER.error(f"[process_row_fmt] Exception occured!")
            tb = traceback.format_exc()
            LOGGER.error(tb)
            row["num_attempts"] += 1


    def process_rows_fmt(self, rows, temperature, key_name="prompt"):
        """
        Move multiple conversations along by one step (e.g., 1->2 or 3->4)

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
        filtered_rows = [row for row in rows if not is_conversation_done(row)]

        # Early exit, if no rows to process
        if not filtered_rows:
            LOGGER.debug("No grouped rows to process")
            return

        model_provider = self.llm_config["model_provider"]
        # CASE 1: If vLLM, pass multiple row prompts at once
        if model_provider == "vllm":
            # Raise error, if choices provided
            if "choices" in filtered_rows[0]:
                raise RuntimeError("Chat template is not compatible with choices currently!")

            # Get conversation till this point
            conversations = []
            conv_turns = []
            for row in filtered_rows:
                curr_turn, conversation = extract_conversation_till_turn(row)
                # Prepend system message
                if self.system_prompt:
                    conversation.insert(0, {"role": "system", "content": self.system_prompt})
                # Raise error, if conversation is already done
                if curr_turn is None:
                    raise RuntimeError("Conversation is already ended, despite being flagged as not done!")
                conversations.append(conversation)
                conv_turns.append(curr_turn)

            # Get LLM responses
            llm_responses = self.vllm_chat_multiple(conversations)

            # Assign responses for each row
            for idx, row in enumerate(filtered_rows):
                # CASE 1: Valid non-empty response
                curr_response_text = llm_responses[idx]["res"]
                response_key = FMT_ASSISTANT_KEY.format(turn=conv_turns[idx])
                row[response_key] = curr_response_text
                # NOTE: Error-handling is commented out for now
                # elif curr_response_text == "":
                #     row["num_attempts"] += 1
                # else:
                #     raise RuntimeError("Unexpected response from vLLM!")
            return

        # CASE 2: If online model, send parallel requests to LLM API
        if is_provider_online(model_provider):
            raise NotImplementedError("Online models with chat format is not yet supported!")
            # with concurrent.futures.ThreadPoolExecutor(max_workers=len(filtered_rows)) as executor:
            #     futures = [
            #         executor.submit(
            #                 self.process_row,
            #                 row=row,
            #                 index=idx,
            #                 temperature=temperature,
            #                 key_name=key_name,
            #             )
            #         for idx, row in enumerate(filtered_rows)
            #     ]

            #     # Wait for all futures to complete
            #     concurrent.futures.wait(futures)
            return

        # TODO: CASE 3: If FastChat, pass multiple prompts sequentially
        raise NotImplementedError()
        return


    def process_file_fmt(self, input_path, output_path, file_config=None):
        """
        Performs inference for a single json file in the FairMT-Bench.

        Args:
            input_path (str): Path to the input data file.
            output_path (str): Path where the processed data will be saved.
            file_config (dict): Configuration settings for file processing.
        """
        # INPUT: Sanitize file_config
        file_config = file_config or {}
        temperature = file_config.get(os.path.basename(input_path), self.llm_config["temperature"])

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
        if all(is_conversation_done(row) for row in saved_data):
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

        # Perform inference in batches
        while not all(is_conversation_done(row) for row in saved_data):
            curr_idx = 0
            grouped_rows = []
            while curr_idx < len(saved_data) and len(grouped_rows) < batch_size:
                row = saved_data[curr_idx]
                # Include if no valid response yet and hasn't failed excessively
                if row.get("num_attempts") < MAX_ATTEMPTS and \
                        not is_conversation_done(row):
                    grouped_rows.append(row)
                curr_idx += 1

            # If no more data, then break before creating threads
            if not grouped_rows:
                break

            # Process the grouped rows
            self.process_rows_fmt(grouped_rows, temperature)

            # Save the updated saved data to the output file
            json_utils.save_json(saved_data, output_path, lock=lock)
            LOGGER.debug(f"Processed {input_path} and saved results to {output_path}")

        # Save the updated saved data to the output file
        json_utils.save_json(saved_data, output_path, lock=lock)
        LOGGER.debug(f"Processed {input_path} and saved results to {output_path}")


    ############################################################################
    #                       Dataset-Level Functions                            #
    ############################################################################
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

        # Temporarilly overwrite parameters, if CEB and closed-ended
        orig_llm_config = self.llm_config.copy()
        if self.dataset_collection == "ceb" and dataset_name in CLOSE_ENDED_DATASETS:
            # Overwrite parameters for close-ended datasets
            LOGGER.info("Overwriting parameters for CEB close-ended datasets (temperature=1, chat template=True)")
            self.llm_config["temperature"] = 1
            self.llm_config["use_chat_template"] = True

        # Specify function to process each file based on dataset collection
        collection_to_proc_func = {
            "ceb": self.process_file_ceb,
            "fmt": self.process_file_fmt,
            "de": self.process_file_ceb,    # NOTE: using choices implemented for CEB
        }
        process_file_func = collection_to_proc_func[self.dataset_collection]

        # Assign dataset-specific system prompt
        self.system_prompt = SYSTEM_PROMPT_MAP[self.system_prompt_type]

        # Creare result dir
        result_dir = os.path.join(DIR_GENERATIONS, self.system_prompt_type, self.model_name, dataset_name)
        os.makedirs(result_dir, exist_ok=True)

        # Process each JSON file
        base_dir = os.path.join(self.data_path, dataset_name)
        json_paths = list(set(glob.glob(os.path.join(base_dir, "*.json"))))
        assert json_paths, f"[LLMGeneration] Could not find data files for dataset `{dataset_name}`"
        for json_path in json_paths:
            LOGGER.debug("Processing file: %s", json_path)
            save_path = os.path.join(result_dir, os.path.basename(json_path))
            process_file_func(json_path, save_path)

        # Revert LLM generation parameters
        self.llm_config = orig_llm_config


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
            LOGGER.error(f"Dataset path {self.data_path} does not exist.")
            return None

        # Iterate for each dataset
        for dataset in self.dataset_names:
            num_retries = 0
            successful = False
            while not successful and num_retries < max_retries:
                try:
                    self.infer_dataset_single(dataset)
                    successful = True
                except Exception as e:
                    LOGGER.error(f"[LLM Inference] Failed on attempt {num_retries + 1}! With the following error trace:")
                    LOGGER.error(traceback.format_exc())
                    LOGGER.error(f"[LLM Inference] Retrying in {retry_interval} seconds...")
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


def extract_model_path_or_name(model_path_or_name, model_provider="vllm", use_chat_template=False):
    """
    Return tuple of model (nick)name and model path, provided either

    Parameters
    ----------
    model_path_or_name : str
        Path to the model, or model (nick)name
    model_provider : str
        Model provider name
    use_chat_template : bool
        If True, use chat template

    Returns
    -------
    tuple of (str, str)
        (i) Model (nick)name
        (ii) Path to model
    """
    # Get model name and path
    model_path_to_name = MODEL_INFO["model_path_to_name"]
    model_name_to_path = {v:k for k,v in model_path_to_name.items()}
    if model_path_or_name in model_path_to_name:
        model_path = model_path_or_name
        model_name = model_path_to_name[model_path_or_name]
    if model_path_or_name in model_name_to_path:
        model_name = model_path_or_name
        model_path = model_name_to_path[model_path_or_name]
    elif model_path_or_name.split("/")[-1] in model_path_to_name:
        model_path = model_path_or_name
        model_name = model_path_to_name[model_path_or_name.split("/")[-1]]
    else:
        raise RuntimeError(
            "Please ensure model path has mapping in `config.py`!"
            f"\n\tModel Path: `{model_path_or_name}`")

    # Ensure model name is valid, if online model is chosen
    if is_provider_online(model_provider):
        assert model_name in MODEL_INFO['online_model'], (
            f"Online model provided `{model_name}` is invalid! "
            f"\nValid options: {MODEL_INFO['online_model']}"
        )

    # If using chat template, append "-chat" to model name
    if use_chat_template:
        model_name += "-chat"

    return model_name, model_path


def apply_chat_template_vllm(llm_engine, prompt, system_prompt=None):
    """
    Apply chat template to text using vLLM

    Parameters
    ----------
    llm_engine : vllm.LLM
        vLLM object
    prompt : str
        User prompt
    system_prompt : str
        Add system prompt, if necessary

    Returns
    -------
    str
        User prompt formatted with chat template
    """
    # Get tokenizer
    tokenizer = llm_engine.get_tokenizer()

    # Add system prompt (if provided) and user prompt
    messages = []
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})
    messages.append({'role': 'user', 'content': prompt})

    # Apply chat template, stored on the tokenizer
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return formatted_prompt


def apply_chat_template_hf(model_path, prompt):
    """
    Apply chat template to text using HuggingFace

    Parameters
    ----------
    model_path : str
        Model path, must contain a tokenizer
    prompt : str
        User prompt

    Returns
    -------
    str
        User prompt formatted with chat template
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Apply chat template
    messages = [{'role': 'user', 'content': prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return formatted_prompt


def apply_chat_template_fastchat(model_path, prompt):
    """
    Apply chat template to text using lm-sys/fastchat

    Parameters
    ----------
    model_path : str
        Model path, must contain a tokenizer
    prompt : str
        User prompt

    Returns
    -------
    str
        User prompt formatted with chat template
    """
    try:
        from fastchat.model import get_conversation_template
    except ImportError:
        raise ImportError("Please `pip install fastchat`!")

    # Get conversation template for model
    conv = get_conversation_template(model_path)
    conv.set_system_message("")
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def compute_prob(vllm_logprobs):
    """
    Compute normalized probability of generated text

    Parameters
    ----------
    vllm_logprobs : list of dict
        List of LogProbs

    Returns
    -------
    float
        Normalized probability of generated text
    """
    try:
        curr_log_probs = [list(l.values())[0].logprob for l in vllm_logprobs]
        seq_log_prob = sum(curr_log_probs) / len(curr_log_probs)
        seq_prob = np.exp(seq_log_prob)
        return seq_prob
    except:
        return None


def extract_choice_logprobs(choice, prompt_logprobs):
    """
    Extract log probabilities for tokens added to the original prompt.

    Parameters
    ----------
    choice : str:
        The hypothetical continuation, whose sequence probability to get.
    prompt_logprobs : list of dict
        Output of output.prompt_logprobs. A list, where each item corresponds
        to each token in the prompt.
    
    Returns:
    Dict containing log probabilities of the added tokens
    """
    # Remove first token, if it's None
    # NOTE: It normally corresponds to the conditional probability of the sequence before it
    if prompt_logprobs[0] is None:
        prompt_logprobs = prompt_logprobs[1:]

    # Work backwards to reconstruct the added text
    added_text = choice
    reconst_choice = ""
    num_prompt_tokens = len(prompt_logprobs)
    for token in prompt_logprobs[::-1]:
        # Find the token with the highest probability (rank 1)
        chosen_token = list(token.values())[0]
        reconst_choice = chosen_token.decoded_token + reconst_choice
        num_prompt_tokens -= 1

        # Stop if we've reconstructed the original prompt. Or if we're already too far
        if added_text.strip() == reconst_choice.strip():
            break

    # Raise error, if we weren't able to extract the prompt
    if choice.strip() != reconst_choice.strip():
        raise RuntimeError(f"""Reconstructed choice doesn't match original choice!

Original: `{choice}`
Reconstructed: `{reconst_choice}`
        """)

    # Get logprobs of continuation tokens
    accum_logprobs = []
    reconst_cont = ""
    for token in prompt_logprobs[num_prompt_tokens:]:
        chosen_token = list(token.values())[0]
        accum_logprobs.append(chosen_token.logprob)
        reconst_cont += chosen_token.decoded_token

    # Compute average log probability
    avg_logprob = sum(accum_logprobs) / len(accum_logprobs)

    # Compute sequence probability
    seq_prob = np.exp(avg_logprob)

    return seq_prob


def is_conversation_done(row):
    return "4-turn Conv Response" in row


def extract_conversation_till_turn(row, turn=4):
    """
    Get conversation until before the latest LLM response.

    Parameters
    ----------
    row : dict
        Row containing a conversation in the form of:
        {
            "0-turn Conv": [USER PROMPT],
            "0-turn Conv Response": [LLM RESPONSE],
            ...
            "4-turn Conv": [USER PROMPT],
            "4-turn Conv Response": [LLM RESPONSE],
        }
    turn : int, optional
        Maximum number of turns to consider

    Returns
    -------
    tuple of (int, list of dict)
        (i) Number of latest turn without a response (or None if complete),
        (ii) Formatted conversation in the form of a user/assistant conversation
        [
            {"role": "user", "content": USER PROMPT (1-turn)},
            {"role": "assistant", "content": LLM RESPONSE (1-turn)},
            ...
            {"role": "user", "content": USER PROMPT (K-turn)},
            # Optional: {"role": "assistant", "content": LLM RESPONSE (K-turn)},
        ]
        , where K is the latest turn without an LLM response 
    """
    accum_conv = []
    for curr_turn in range(0, turn+1):
        # 1. Add prompt
        curr_prompt_key = FMT_USER_KEY.format(turn=curr_turn)
        curr_prompt = row[curr_prompt_key]
        accum_conv.append({
            "role": "user",
            "content": curr_prompt,
        })

        # Check if response is here, if not, return early with turn
        curr_response_key = FMT_ASSISTANT_KEY.format(turn=curr_turn)
        if curr_response_key not in row:
            return curr_turn, accum_conv

        # 2. Add response
        curr_response = row[curr_response_key]
        accum_conv.append({
            "role": "assistant",
            "content": curr_response,
        })

    return None, accum_conv
