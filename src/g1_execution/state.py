import contextlib

class Executor:

    def state(self):
        with contextlib.redirect_stdout(self.null):
            return self.rt.read_state()
