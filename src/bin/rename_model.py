"""
rename_model.py

Description: Command-line script to rename a model name / HuggingFace repo to
             one of the manually specified repository names
"""

# Custom libraries
from config import MODEL_INFO


def rename_model(model_path_or_name):
    """
    Rename model name to explicitly given shorthand.

    Parameters
    ----------
    model_path_or_name : str
        Local/HF Path to Model, or Model (Nick)name

    Returns
    -------
    str
        Model nickname
    """
    # Get model name and path
    model_path_to_name = MODEL_INFO["model_path_to_name"]
    model_name_to_path = {v:k for k,v in model_path_to_name.items()}
    if model_path_or_name in model_path_to_name:
        model_name = model_path_to_name[model_path_or_name]
    if model_path_or_name in model_name_to_path:
        model_name = model_path_or_name
    elif model_path_or_name.split("/")[-1] in model_path_to_name:
        model_name = model_path_to_name[model_path_or_name.split("/")[-1]]
    else:
        raise RuntimeError(
            "Please ensure model path has mapping in `config.py`!"
            f"\n\tModel Path: `{model_path_or_name}`")
    return model_name


if __name__ == "__main__":
    from fire import Fire
    Fire(rename_model)
