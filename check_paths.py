from pathlib import Path
import pandas as pd

# Candidate roots (OneDrive personal vs org installs can differ)
roots = [
    Path(r"C:\Users\arshp\OneDrive\Desktop\Terra Research"),
    Path(r"C:\Users\arshp\OneDrive - Personal\Desktop\Terra Research"),
    Path(r"C:\Users\arshp\Desktop\Terra Research"),
    Path(r"C:\Users\arshp"),
]

flight_file = None
for root in roots:
    if root.exists():
        # Try common variations: CSV/XLSX, hyphen/en-dash, spacing
        patterns = [
            "*JFK-LAX*Model*.csv",
            "*JFK–LAX*Model*.csv",     # en dash U+2013
            "*JFK - LAX*Model*.csv",
            "*JFK*LAX*Model*.csv",
            "*JFK-LAX*Model*.xlsx",
            "*JFK–LAX*Model*.xlsx",
            "*JFK*LAX*Model*.xlsx",
        ]
        for pat in patterns:
            for p in root.rglob(pat):
                flight_file = p
                break
        if flight_file:
            break

if not flight_file:
    raise FileNotFoundError("Could not locate the flight file under likely roots. "
                            "Run the search script to see exact name & location.")

print("Reading:", flight_file)
flight_data = (pd.read_excel(flight_file) if flight_file.suffix.lower() in {".xlsx", ".xls"}
               else pd.read_csv(flight_file))
