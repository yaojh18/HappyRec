from .FM import FM

class AFM(FM):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_modules(self, *args, **kwargs) -> None:
        pass

    def forward(self, batch, *args, **kwargs):
        pass