import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = [["run_params.py"], ["theory.py"], ["bench.py"], ["latency.py"], ["extra.py"], ["lattice_attack.py"],
         ["figures.py"], ["tables.py"], ["tables.py", "appendix"]]


def main():
    for step in STEPS:
        print("running", " ".join(step), flush=True)
        subprocess.run([sys.executable, os.path.join(HERE, step[0])] + step[1:], check=True)


if __name__ == "__main__":
    main()
