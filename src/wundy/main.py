import sys
import argparse
import tkinter as tk
from tkinter import filedialog
import wundy
from wundy import ui
from wundy import first


def pick_yaml_file() -> str:
    
    root = tk.Tk()
    root.withdraw()  

    path = filedialog.askopenfilename(
        title="Select Wundy YAML File",
        filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")]
    )

    root.destroy()

    if not path:
        print("No file selected. Exiting.")
        sys.exit(1)

    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "file",
        nargs="?",
        help="Wundy YAML file (optional — if omitted, a file picker will open)"
    )
    args = p.parse_args()

    yaml_file = args.file if args.file else pick_yaml_file()

    with open(yaml_file) as fh:
        data = ui.load(fh)

    inp = wundy.ui.preprocess(data)
    soln = first.first_fe_code(
        inp["coords"],
        inp["blocks"],
        inp["bcs"],
        inp["dload"],
        inp["materials"],
        inp["block_elem_map"],
    )

    print(soln)


if __name__ == "__main__":
    sys.exit(main())
