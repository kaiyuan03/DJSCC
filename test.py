import pickle
import types
a = 1

def save_variables(filename):
    with open(filename, 'wb') as f:
        # Filter out unpickleable objects
        variables = {k: v for k, v in globals().items() if not k.startswith('__') and not isinstance(v, types.ModuleType) and not isinstance(v, types.FunctionType)}
        pickle.dump(variables, f)

def load_variables(filename):
    with open(filename, 'rb') as f:
        variables = pickle.load(f)
        globals().update(variables)
save_variables("test.txt")