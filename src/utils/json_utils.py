# Standard libraries
import json
import logging


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
    # Acquire lock
    if lock is not None:
        lock.acquire()

    # Write to file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    # Release lock
    if lock is not None:
        lock.release()


def update_with_existing_data(
        new_data,
        prev_data=None,
        prev_path=None,
        rename_keys=None,
        prompt_key="prompt",
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
    prompt_key : str, optional
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

    # Get mapping of prompt to prev. row
    prompt_to_old = {
        row[prompt_key]: row
        for row in prev_data
    }
    # For each row in the current dataset, attempt to load in
    # whatever was created from the previous session
    for row in new_data:
        curr_prompt = row[prompt_key]
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
