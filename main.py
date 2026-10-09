import os
import subprocess
import sys

def run_script(script_path):
    print(f"\n{'*'*50}\nRunning {script_path}\n{'*'*50}")
    result = subprocess.run([sys.executable, script_path])
    if result.returncode != 0:
        print(f"Error running {script_path}")
        sys.exit(result.returncode)
    print(f"Finished {script_path}\n")

if __name__ == "__main__":
    print("Starting ML Pipeline...\n")
    
    scripts = [
        os.path.join("src", "part1_eda.py"),
        os.path.join("src", "part2_preprocessing.py"),
        os.path.join("src", "part3_kmeans_clustering.py")
    ]
    
    for script in scripts:
        if not os.path.exists(script):
            print(f"Error: Could not find {script}")
            sys.exit(1)
            
        run_script(script)
        
    print("Pipeline completed successfully.")
