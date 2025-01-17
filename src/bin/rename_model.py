"""
rename_model.py

Description: Command-line script to rename a model name / HuggingFace repo to
             one of the manually specified repository names
"""

# Custom libraries
from config import MODEL_INFO


def rename_model(model_path_or_name, reverse=False):
    """
    Rename model name to explicitly given shorthand. If reverse, then give model
    name/path from shorthand

    Parameters
    ----------
    model_path_or_name : str
        Local/HF Path to Model, or Model (Nick)name.

    Returns
    -------
    str
        Model nickname if not `reverse` else return model path
    """
    # Get model name and path
    model_path_to_name = MODEL_INFO["model_path_to_name"]
    model_name_to_path = {v:k for k,v in model_path_to_name.items()}
    # CASE 1: Model path provided
    if model_path_or_name in model_path_to_name:
        return model_path_or_name if reverse else model_path_to_name[model_path_or_name]
    # CASE 2: Model name provided
    elif model_path_or_name in model_name_to_path:
        return model_name_to_path[model_path_or_name] if reverse else model_path_or_name
    # CASE 3: Model path provided, but only part of name is valid
    elif model_path_or_name.split("/")[-1] in model_path_to_name:
        model_name = model_path_to_name[model_path_or_name.split("/")[-1]]
        return model_path_or_name if reverse else model_name

    # CASE 4: Invalid name provided
    raise RuntimeError(
        "Please ensure model path has mapping in `config.py`!"
        f"\n\tModel Path: `{model_path_or_name}`")


if __name__ == "__main__":
    from fire import Fire
    Fire(rename_model)
