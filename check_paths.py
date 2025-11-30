import os
import sys

def list_files(startpath):
    print(f"Scanning directory: {startpath}")
    if not os.path.exists(startpath):
        print(f"ERROR: Path does not exist: {startpath}")
        return

    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        # Limit file printing to avoid huge logs
        for f in files[:5]:
            print(f"{subindent}{f}")
        if len(files) > 5:
            print(f"{subindent}... ({len(files)-5} more files)")

if __name__ == "__main__":
    base_dir = "./data/prepared"
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    
    list_files(os.path.abspath(base_dir))