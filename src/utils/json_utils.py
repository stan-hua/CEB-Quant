# Standard libraries
import json
import logging
import os
import tempfile


################################################################################
#                                  Constants                                   #
################################################################################
LOGGER = logging.getLogger(__name__)


################################################################################
#                               Helper Functions                               #
################################################################################
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, file_path, lock=None):
    """
    Saves data to a JSON file using a temporary file and atomic rename,
    ensuring the lock is released and temporary files are cleaned up
    even if a KeyboardInterrupt or other error occurs.

    Parameters
    ----------
    data : list of dict
        Data to save
    file_path : str
        Path to the output file
    lock : threading.Lock
        Lock to use for synchronization
    """
    temp_file_path = None # Initialize temp_file_path to None

    # Acquire lock if provided
    if lock is not None:
        lock.acquire()
        print("Lock acquired.") # Added for demonstration

    try:
        # Determine the directory of the target file
        output_dir = os.path.dirname(file_path)
        # If no directory is specified, use the current directory
        if not output_dir:
            output_dir = '.'

        # Create a temporary file in the same directory as the output file.
        # Using delete=False means we are responsible for deleting the file.
        # This is necessary because we need to close the file before renaming it.
        with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=output_dir, encoding='utf-8') as tmp_file:
            temp_file_path = tmp_file.name
            print(f"Writing data to temporary file: {temp_file_path}...")
            json.dump(data, tmp_file, ensure_ascii=False, indent=4)
            # The 'with' statement ensures the temporary file is closed here.

        # Atomically rename the temporary file to the final destination.
        # This replaces the original file if it exists.
        print(f"Renaming temporary file to final destination: {file_path}...")
        os.replace(temp_file_path, file_path) # os.replace is atomic
        print("Data saved successfully.")

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected! Cleaning up...")
        # The finally block will execute next, releasing the lock and cleaning up temp file.
        raise # Re-raise the interrupt so the program exits

    except Exception as e:
        print(f"An error occurred during file saving: {e}")
        # The finally block will execute next, releasing the lock and cleaning up temp file.
        raise # Re-raise the exception

    finally:
        # Release lock if provided and if it was acquired
        if lock is not None:
            lock.release()
            print("Lock released.")

        # Clean up the temporary file if it still exists (e.g., if rename failed)
        if temp_file_path and os.path.exists(temp_file_path):
            print(f"Cleaning up temporary file: {temp_file_path}")
            try:
                os.remove(temp_file_path)
            except OSError as e:
                print(f"Error removing temporary file {temp_file_path}: {e}")


def update_with_existing_data(
        new_data,
        prev_data=None,
        prev_path=None,
        rename_keys=None,
        prompt_col="prompt",
    ):
    """
    Given current data and saved (evaluation) data from a previous run, update
    the current data to sync data from the previous run.

    Parameters
    ----------
    new_data : list of dict
        List of rows, where each row is a question/response
    prev_data : list of dict
        List of rows, where each row is a question/response
    prev_path : str
        Path to JSON file containing saved data from a previous run, which should
        correspond to data from `new_data`
    rename_keys : dict
        If provided, renames existing keys
    prompt_col : str, optional
        Name of prompt key

    Returns
    -------
    list of dict
        New data where rows that exist in `prev_path` are used to update the row
    """
    assert prev_path or prev_data, "One of `prev_path` or `prev_data` must be provided!"

    # Attempt to load previous data
    if prev_path:
        try:
            prev_data = load_json(prev_path)
        except FileNotFoundError:
            LOGGER.warning("No saved progress file found at %s. Starting a new evaluation.", prev_path)
            return new_data

    # Early return, if previous data is empty
    if not prev_data:
        return new_data

    # Log if size changed
    # CASE 1: Length of current data and previous data is the same
    if len(new_data) == len(prev_data):
        LOGGER.info("Resuming from saved progress.")
    # CASE 2: Length of current data and previous data is different
    else:
        LOGGER.info("Size changed since previous save! Attempting to salvage saved progress.")

    # If index column exists, use that to synchronize updates
    if "idx" in new_data[0] and "idx" in prev_data[0]:
        prompt_col = "idx"

    # Get mapping of prompt to prev. row
    prompt_to_old = {
        row[prompt_col]: row
        for row in prev_data
    }
    # For each row in the current dataset, attempt to load in
    # whatever was created from the previous session
    for row in new_data:
        curr_prompt = row[prompt_col]
        if curr_prompt not in prompt_to_old:
            continue
        prev_row = prompt_to_old[curr_prompt].copy()
        if rename_keys:
            for old_key, new_key in rename_keys.items():
                if old_key in prev_row:
                    prev_row[new_key] = prev_row.pop(old_key)
        row.update(prev_row)
    return new_data


def update_nested_dict(data, *keys, value=None):
    """
    Updates a nested dictionary, given a variable number of keys.

    Parameters
    ----------
    data : dict
        The dictionary to update
    *keys : list of str
        List of keys to traverse, in order
    value : obj
        The value to assign to the last key in `keys`

    Returns
    -------
    dict
        The updated dictionary
    """
    curr_dict = data
    for idx, key in enumerate(keys):
        # CASE 1: Last layer, assign value
        if idx == len(keys) - 1:
            curr_dict[key] = value
            continue
        # CASE 2: Intermediate layer, and key doesn't exist
        if key not in curr_dict:
            curr_dict[key] = {}
        curr_dict = curr_dict[key]
    return data
