import os
from huggingface_hub import hf_hub_download
import hf_transfer


class AutoDownloadModels:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "repo_id": ("STRING", {"tooltip": "Enter hugging face repo id"}),
                "model_name": ("STRING", {"tooltip": "Enter hugging face model name to download"}),
                "path_to_download": ("STRING", {"tooltip": "Enter to download the model to"}),
            },
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "DownloadModel"
    CATEGORY = "DownloadModel"
    DESCRIPTION = "This node downloads models from huggingface models"

    def DownloadModel(self, repo_id, model_name, path_to_download):
            model_path = hf_hub_download(repo_id=repo_id, filename=model_name, local_dir=path_to_download)

            return model_path

         
        

NODE_CLASS_MAPPINGS = {
    "AutoDownloadModels": AutoDownloadModels,

    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoDownloadModels": "Auto Download Models",
    }
