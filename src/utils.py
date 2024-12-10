import json
from typing import Union

class style():
  RED = '\033[31m'
  GREEN = '\033[32m'
  YELLOW = '\033[33m'
  BLUE = '\033[34m'
  CYAN = '\033[35m'
  PURPLE = '\033[36m'
  RESET = '\033[0m'  # Your shell's stdout default
  
class Struct():
    def __init__(self, **entries):
        self.config_dict = entries
        for key, value in entries.items():
            setattr(self, key, value)

    def __str__(self):
        s = "Struct: {"
        for key, value in self.config_dict.items():
            s += f"{key}: {value},"
        s = s[:-1] + "}"
        return s

    def get_config_dict(self):
        return self.config_dict
    
def get_context_window(llm)->Union[int, None]:

    if 'transformer.' in str(type(llm)).lower():
        typical_fields = ["max_position_embeddings", "n_positions", "seq_len", "seq_length", "n_ctx", "sliding_window"]  
        context_windows = [getattr(llm.config, field) for field in typical_fields if field in dir(llm.config)]

        max_len = (context_windows.pop()) if len(context_windows) else None
        if max_len==None:
            print(f"No context length variable found for model.")

    elif 'ollama' in str(type(llm)).lower():
        max_len = json.loads(llm.json())['context_window']

    else:
        # raise Exception("Model must belong to Transformers or Ollama libraries to get context window.")
        print(str(type(llm)).lower())
        max_len = 1024 # Default value for BERT-like models
    return max_len