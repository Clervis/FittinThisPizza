import csv
import random
from datetime import datetime, timedelta

# Output file
filename = "fitness_data.csv"

# Header
header = [
    "Christian Weight",
    "Krysty Weight",
    "Christian Target",
    "Krysty Target",
    "Date"
]

# Starting values
christian_target = 263
krysty_target = 138

start_date = datetime.strptime("02-14-2026", "%m-%d-%Y")
end_date = datetime.strptime("12-31-2026", "%m-%d-%Y")
cutoff_date = datetime.strptime("05-17-2026", "%m-%d-%Y")

# Daily target deltas
christian_delta_early = -(2.3/7)
christian_delta_late = -(1/7)
krysty_delta = -(18/318)

# Jitter ranges (only used for first 60 days)
christian_jitter_range = (-0.4, 0.4)
krysty_jitter_range = (-0.25, 0.25)

# Write CSV
with open(filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(header)

    current_date = start_date
    day_count = 0

    while current_date <= end_date:
        date_str = current_date.strftime("%m-%d-%Y")

        # Apply jitter only for first 60 days
        if day_count < 60:
            christian_jitter = random.uniform(*christian_jitter_range)
            krysty_jitter = random.uniform(*krysty_jitter_range)
            christian_weight = round(christian_target + christian_jitter, 3)
            krysty_weight = round(krysty_target + krysty_jitter, 3)
        else:
            christian_weight = ""
            krysty_weight = ""

        # Write row
        writer.writerow([
            christian_weight,
            krysty_weight,
            round(christian_target, 3),
            round(krysty_target, 3),
            date_str
        ])

        # Update targets
        if current_date <= cutoff_date:
            christian_target += christian_delta_early
        else:
            christian_target += christian_delta_late

        krysty_target += krysty_delta

        # Advance date
        current_date += timedelta(days=1)
        day_count += 1

print(f"{filename} created with targets through Dec 31, 2026 and weights for first 60 days.")
