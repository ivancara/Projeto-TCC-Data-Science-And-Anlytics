from utils.constants.AppConfig import AppConfig
class ConstantsManagement:
    def __init__(self):
        # Set constants from separate classes as attributes
        for cls in [AppConfig]:
            for key, value in cls.__dict__.items():
                if not key.startswith("__"):
                    self.__dict__.update(**{key: value})

    def __setattr__(self, name, value):
        raise TypeError("Constants are immutable")