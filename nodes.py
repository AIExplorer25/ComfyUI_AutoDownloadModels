import os
from huggingface_hub import hf_hub_download
import hf_transfer
import json
from rapidfuzz import process, fuzz
from collections import Counter
from huggingface_hub import list_models
from huggingface_hub import HfApi
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

            return (model_path,)

class AutoDownloadALLModels:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "modelandpath": ("STRING", {"tooltip": "Enter hugging face repo id"}),
                 "repo_ids_manual": ("STRING", {"tooltip": "Enter hugging face repo id"}),
            },
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "DownloadALLModel"
    CATEGORY = "DownloadALLModel"
    DESCRIPTION = "This node downloads models from huggingface models"

    def DownloadALLModel(self, modelandpath, repo_ids_manual):
            manualrepoids=repo_ids_manual.split(",")
            split_text = modelandpath.strip("()").split("),(")
            status=[]
            for item in split_text:
                try:
                    print(item)
                    model_name, folder_name = item.split("~~~")
                    mname, extension = model_name.rsplit('.', 1)
                    repo_ids=self.search_huggingface_models(mname,limit=5)
                    for repo in repo_ids:
                        status.append(repo)
                    repo_id=""
                    if(len(repo_ids)>0):
                        repo_id=repo_ids[0]
                        model_file_name=self.search_file_in_repo(repo_id,model_name)
                        model_path = hf_hub_download(repo_id=repo_id, filename=model_file_name, local_dir="/workspace/ComfyUI/models/"+folder_name+"/")
                        status.append(model_path)
                    else:
                        for manualrepo in manualrepoids:
                            model_file_name=self.search_file_in_repo(manualrepo,model_name)
                            if(model_file_name!=None):
                                model_path = hf_hub_download(repo_id=manualrepo, filename=model_file_name, local_dir="/workspace/ComfyUI/models/"+folder_name+"/")
                                status.append(model_path)
                                break
                            

                    
         
                    

                except Exception as e:
                    print(f"Error: Cannot process '{item}'. Exception: {e}")
                    status.append(f"Error: Cannot process '{item}'. Exception: {e}")

            return (", ".join(status),)         

    def search_file_in_repo(self,repo_id, filename):
        """
        Searches for a specific file inside a Hugging Face repository.

        Args:
            repo_id (str): The Hugging Face repo ID (e.g., "gemasai/4x_NMKD-Siax_200k").
            filename (str): The filename to search for.

        Returns:
            str or None: The file path if found, otherwise None.
        """
        api = HfApi()
        files = api.list_repo_files(repo_id)

        for file in files:
            if filename in file:
                return file  # Return the matching file path

        return None  # Return None if file is not found
    def search_huggingface_models(self, query, limit=10):
        """
        Searches for models on Hugging Face based on the query.

        Args:
            query (str): The search keyword.
            limit (int): Maximum number of results to return.

        Returns:
            list: A list of matching model names.
        """
        models = list_models(search=query, limit=limit)
        #model=models[0]
        #return model.id
        return [model.id for model in models]  # Extract only model IDs   
class GetModelsFromWorkflow:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "workflow_path": ("STRING", {"tooltip": "Enter path of your workflow"}), 
                "models_folder_path": ("STRING", {"default":"/workspace/ComfyUI/models/", "tooltip": "Enter path of your workflow"}),
            },
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("pa",)
    FUNCTION = "find_ml_models"
    CATEGORY = "DownloadModel"
    DESCRIPTION = "This node maps models from huggingface to folders"





    def get_folders_list(self, directory_path):
        """
        Returns a list of folder names inside the given directory.

        Args:
            directory_path (str): The path to the directory.

        Returns:
            list: A list containing folder names.
        """
        try:
            return [f for f in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, f))]
        except FileNotFoundError:
            print("Error: Directory not found.")
            return []
        except PermissionError:
            print("Error: Permission denied.")
            return []

    def find_ml_models(self, workflow_path, models_folder_path):
        """
        Finds all ML models inside a JSON file that belong to predefined folder categories.

        Args:
            json_path (str): Path to the JSON file.
            base_directory (str): Path to the base directory where models might be stored.

        Returns:
            tuple: (list of model files, list of nodes containing models)
        """
        model_extensions = {
            ".safetensors", ".pth", ".pkl", ".bin", ".onnx", ".h5", ".ckpt", ".tflite",
            ".pb", ".joblib", ".pt", ".tar", ".npz", ".mar", ".npy", ".mlmodel",
            ".hdf5", ".weights", ".model", ".caffemodel", ".pbtxt"
        }

        valid_folders = {'checkpoints', 'controlnet', 'gligen', 'style_models', 'vae', 'clip', 
                         'diffusers', 'hypernetworks', 'text_encoders', 'vae_approx', 'clip_vision', 
                         'diffusion_models', 'loras', 'unet', 'ckpt', 'configs', 'embeddings', 
                         'photomaker', 'upscale_models'}

        existing_folders = set(self.get_folders_list(models_folder_path))

        model_files = []
        model_nodes = []

        with open(workflow_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for node in data.get("nodes", []):
            properties = node.get("properties", {})

            if "models" in properties and isinstance(properties["models"], list):
                for model in properties["models"]:
                    if isinstance(model, dict) and "name" in model:
                        model_name = model["name"]
                        if any(model_name.endswith(ext) for ext in model_extensions):
                            model_path = model.get("directory", "")
                            if model_path in valid_folders and model_path in existing_folders:
                                model_files.append(model_name)
                                model_nodes.append(node)

            if "widgets_values" in node:
                for value in node["widgets_values"]:
                    if isinstance(value, str) and any(value.endswith(ext) for ext in model_extensions):
                        model_files.append(value)
                        model_nodes.append(node)

        model_mappings = []

        for model, node in zip(model_files, model_nodes):
            node_type = node.get("type", "").lower()

            # Default to 'checkpoints' if no match is found
            probable_folder = "checkpoints"
            nodecontent=json.dumps(node, ensure_ascii=False).lower()
            probable_folder_percentage=self.calculate_partial_word_percentage(valid_folders, nodecontent, threshold=80)

            probable_folder=probable_folder_percentage[0]

            model_mappings.append(f"("+model+ "~~~"+ probable_folder+")")
            
        print(len(model_mappings))
        formatted_string = ",".join(model_mappings) 
        print(formatted_string)    
        return (formatted_string,)



    def calculate_partial_word_percentage(self, valid_folders, big_text, threshold=80):
        """
        Calculates the percentage of each word in valid_folders that appears fully or partially in big_text.

        Args:
            valid_folders (set): A set of words to check.
            big_text (str): The large text to analyze.
            threshold (int): The similarity threshold (default: 80).

        Returns:
            dict: Dictionary with words as keys and their occurrence percentage as values.
        """
        big_text_lower = big_text.lower().split()  # Convert text to lowercase and split into words
        total_words = len(big_text_lower)  # Total words in the big text

        word_percentages = {}

        # Check each folder name for full or partial presence
        for word in valid_folders:
            matches = process.extract(word.lower(), big_text_lower, scorer=fuzz.partial_ratio, limit=None)
            match_count = sum(1 for match in matches if match[1] >= threshold)  # Count words with high similarity

            # Calculate percentage
            word_percentages[word] = (match_count / total_words) * 100 if total_words > 0 else 0
        best_match = max(word_percentages.items(), key=lambda x: x[1], default=(None, 0))
        return best_match





         
class ShowModelsAndFolderMappings:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mappings": ("STRING",),
            },
        }

    FUNCTION = "ShowMappings"
    CATEGORY = "ShowMappings"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Models Paths",)
    DESCRIPTION = "This node downloads models from huggingface models"

    def ShowMappings(self, mappings):
        print(len(mappings))
        formatted_list = []
        for mapping in mappings:
            print("H-------------------------------")
            print(mapping)
            formatted_item = "("+mapping+")"
            formatted_list.append(formatted_item)
        formatted_string = ",".join(formatted_list)
        print("Helloooooooooo2")
        print(formatted_string)
        return formatted_string

     


 
class SetModelPath:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "modelandpath": ("STRING", {"tooltip": "Enter hugging face repo id"}),
            },
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("modelpath",)
    FUNCTION = "SetModelPath"
    CATEGORY = "SetModelPath"
    DESCRIPTION = "This node downloads models from huggingface models"

    def SetModelPath(self, modelandpath):
           

            return (modelandpath,)

NODE_CLASS_MAPPINGS = {
    "AutoDownloadModels": AutoDownloadModels,
     "ShowModelsAndFolderMappings": ShowModelsAndFolderMappings,
      "GetModelsFromWorkflow": GetModelsFromWorkflow,
      "SetModelPath": SetModelPath,
      "AutoDownloadALLModels": AutoDownloadALLModels,
      
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoDownloadModels": "Auto Download Models",
    "ShowModelsAndFolderMappings": "Show Models And Folder Mappings",
    "GetModelsFromWorkflow": "Get Models From Workflow",
    "SetModelPath": "Set Model Path",
    "AutoDownloadALLModels": "Auto Download ALL Models",
    }
