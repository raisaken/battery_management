import matplotlib.pyplot as plt
import pandas as pd

from config import PROCESSED_FILE, FIGURES_DIR
from utils import ensure_directories


def main():
    ensure_directories([FIGURES_DIR])
    df = pd.read_csv(PROCESSED_FILE)

    # Capacity vs cycle
    plt.figure(figsize=(9, 6))
    for battery_id, group in df.groupby("battery_id"):
        plt.plot(group["cycle_number"], group["capacity_ah"], label=battery_id)
    plt.xlabel("Discharge cycle")
    plt.ylabel("Capacity (Ah)")
    plt.title("NASA Battery Capacity Degradation")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = FIGURES_DIR / "capacity_vs_cycle.png"
    plt.savefig(out, dpi=180)
    plt.close()

    # SOH vs cycle
    plt.figure(figsize=(9, 6))
    for battery_id, group in df.groupby("battery_id"):
        plt.plot(group["cycle_number"], group["soh_percent"], label=battery_id)
    plt.axhline(70, linestyle="--", linewidth=1, label="70% EOL threshold")
    plt.xlabel("Discharge cycle")
    plt.ylabel("SOH (%)")
    plt.title("State of Health Degradation")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out2 = FIGURES_DIR / "soh_vs_cycle.png"
    plt.savefig(out2, dpi=180)
    plt.close()

    # Temperature vs cycle
    plt.figure(figsize=(9, 6))
    for battery_id, group in df.groupby("battery_id"):
        plt.plot(group["cycle_number"], group["avg_temperature"], label=battery_id)
    plt.xlabel("Discharge cycle")
    plt.ylabel("Average measured temperature (°C)")
    plt.title("Average Discharge Temperature by Cycle")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out3 = FIGURES_DIR / "temperature_vs_cycle.png"
    plt.savefig(out3, dpi=180)
    plt.close()

    print("EDA figures saved:")
    print(out)
    print(out2)
    print(out3)


if __name__ == "__main__":
    main()
