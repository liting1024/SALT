import os
import sys
import importlib

def get_latest_file(directory):
    if not os.path.exists(directory):
        raise ValueError(f"The directory {directory} does not exist.")
    files = [os.path.join(directory, file) for file in os.listdir(directory)]
    files = [file for file in files if os.path.isfile(file)]
    if not files:
        return None
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def dynamic_import_lib(library):
    last_dot = library.rfind('.')
    module_name, class_name = library[:last_dot], library[last_dot+1:]
    if 'Baseline' in module_name:
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    module = importlib.import_module(module_name)
    return getattr(module, class_name) 