import os
import subprocess
import sys
import argparse

def install_all():
    path = os.path.dirname(os.path.abspath(__file__))
    packages = [p for p in next(os.walk(path))[1]
                if "setup.py" in os.listdir(p)]

    subprocess.call([sys.executable, "-m", "pip", "install"] +
            sum([["-e", package] for package in packages], []))

def uninstall_all():
    path = os.path.dirname(os.path.abspath(__file__))
    packages = [p for p in next(os.walk(path))[1]
                if "setup.py" in os.listdir(p)]

    eggs = sum([[p+"/"+f for f in os.listdir(p) if "egg-info" in f] for p in packages], [])

    for egg in eggs:
        basename = egg.split(".")[0].split("/")[-1]
        subprocess.call([sys.executable, f"-m", "pip", "uninstall", "-y", basename])
        subprocess.call(["rm", "-r", egg])

parser = argparse.ArgumentParser(description="Install/uninstall all thesis packages "
                                             "in editable mode")

parser.add_argument("--install", default=False, action="store_true")
parser.add_argument("--uninstall", default=False, action="store_true")

if __name__ == "__main__":
    args = parser.parse_args()
    if not (args.install or args.uninstall):
        args.install = True

    if args.install and args.uninstall:
        raise ValueError("Can't install & uninstall")

    if args.install:
        install_all()
    elif args.uninstall:
        uninstall_all()
