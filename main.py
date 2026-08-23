import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(script_name):
    script = ROOT / "src" / script_name
    print("\n" + "=" * 80)
    print(f"RUNNING: {script_name}")
    print("=" * 80)
    subprocess.run([sys.executable, str(script)], check=True)


def main():
    run("check_setup.py")
    run("preprocess.py")
    run("plot_eda.py")
    run("train_soh.py")
    run("train_rul.py")

    print("\nPipeline complete.")
    print("Check:")
    print("  data/processed/battery_cycles.csv")
    print("  results/figures/")
    print("  results/metrics/")
    print("  models/")


if __name__ == "__main__":
    main()
