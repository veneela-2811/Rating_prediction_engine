import subprocess
import sys
from datetime import datetime

# ================== Configuration ==================
embedding_sizes = [16, 32, 48, 64, 128]
training_script = "main.py"  # your training script
test_script = "test.py"       # your masked evaluation script
log_file = "embedding_sweep_results.txt"

# ================== Open log file ==================
with open(log_file, "w") as f:
    f.write(f"Embedding Sweep Results - {datetime.now()}\n")
    f.write("="*60 + "\n\n")

# ================== Run training and testing ==================
for emb_size in embedding_sizes:
    print(f"\nRunning embedding dimension: {emb_size}")
    with open(log_file, "a") as f:
        f.write(f"Embedding dimension: {emb_size}\n")
        f.write("-"*40 + "\n")
    
    # Run training script with embedding size as argument
    train_proc = subprocess.run(
        [sys.executable, training_script, str(emb_size)],
        capture_output=True,
        text=True
    )
    
    # Write training output to log
    with open(log_file, "a") as f:
        f.write("=== Training Output ===\n")
        f.write(train_proc.stdout + "\n")
        if train_proc.stderr:
            f.write("=== Training Errors ===\n")
            f.write(train_proc.stderr + "\n")
    
    # Run masked test evaluation
    test_proc = subprocess.run(
        [sys.executable, test_script, f"recommendation_model_{emb_size}.pkl"],
        capture_output=True,
        text=True
    )
    
    # Write test output to log
    with open(log_file, "a") as f:
        f.write("=== Masked Test Output ===\n")
        f.write(test_proc.stdout + "\n")
        if test_proc.stderr:
            f.write("=== Test Errors ===\n")
            f.write(test_proc.stderr + "\n")
    
    with open(log_file, "a") as f:
        f.write("\n" + "="*60 + "\n\n")

print(f"All runs completed. Results saved to {log_file}")
