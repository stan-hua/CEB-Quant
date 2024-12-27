"""
rename_model.py

Description: Command-line script to rename a model name / HuggingFace repo to
             one of the manually specified repository names
"""

# Custom libraries
import config


def rename_model(model_name):
    model_mapping = config.MODEL_INFO['model_mapping']
    if model_name not in model_mapping:
        raise ValueError(f"Model name {model_name} not found in model mapping.")
    return model_mapping[model_name]


if __name__ == "__main__":
    from fire import Fire
    Fire(rename_model)
