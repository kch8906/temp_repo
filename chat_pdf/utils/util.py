import os
from typing import List
    
def create_Directory(directory: str):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")
        
def check_Vectorstore(directory: str) -> List[str]:
    return os.listdir(directory)