"""
FreeU Setting parameters.
Refer to https://github.com/ChenyangSi/FreeU
"""

class BaseFreeUParameter:
    b1 = 1.2
    b2 = 1.4
    s1 = 0.9
    s2 = 0.2

class FreeUParameterSD14(BaseFreeUParameter):
    b1: float = 1.2
    b2: float = 1.4
    s1: float = 0.9
    s2: float = 0.2

class FreeUParameterSD21(BaseFreeUParameter):
    b1: float = 1.1
    b2: float = 1.2
    s1: float = 0.9
    s2: float = 0.2

class ControllableFreeUParameter(BaseFreeUParameter):
    def __init__(self, b1:float = 1.2, b2:float = 1.4, s1:float = 0.9, s2:float = 0.2):
        self.b1 = b1
        self.b2 = b2
        self.s1 = s1
        self.s2 = s2
        # Recommended as range 1~1.2 (b1), 1.2~1.6 (b2), s1 <=1, s2 <=1
