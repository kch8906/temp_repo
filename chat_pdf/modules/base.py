from abc import ABCMeta, abstractmethod
from utils.preprocessing import module_parser

class BaseModule(metaclass=ABCMeta):
    def __init__(self,
                 yaml_data = None,
                 module_type: str = None,
                 documents = None,
                 qa_data = None):
        self.documents = documents
        self.qa_data = qa_data
        self.module_config = module_parser(yaml_data, module_type)
    
    @abstractmethod
    def invoke(self):
        pass
    
    @abstractmethod
    def score(self):
        pass