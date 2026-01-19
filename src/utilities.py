import time

# empty context manager so as to not repeat oneself in epoch loop
class NullContext:
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc, tb):
        return False


class Timer:
    def __init__(self, title, show=True):
        self.title = title
        self.show = show
    
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc, tb):
        self.timer = time.perf_counter() - self.start
        if self.show:
            print(f"Elapsed time for block [{self.title}]: {self.timer}s")

        return False
