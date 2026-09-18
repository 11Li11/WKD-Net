class Registry:
    def __init__(self):
        self.modules = {}

    def register_module(self, function):
        self.modules[function.__name__] = function
        return function


MODEL = Registry()
