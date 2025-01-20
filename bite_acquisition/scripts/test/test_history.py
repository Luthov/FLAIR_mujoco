import ast
import os
import sys
from contextlib import redirect_stdout

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)

    def flush(self):
        for f in self.files:
            f.flush()

script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'history.txt')
output_path = os.path.join(script_dir, 'output.txt')

with open(output_path, "w") as f, redirect_stdout(Tee(sys.stdout, f)):
    with open(file_path, 'r') as f:
        bite_history = ast.literal_eval(f.read().strip())

    print("test sys.stdout")
    print(type(bite_history[0][0]))
    print("LMAO")