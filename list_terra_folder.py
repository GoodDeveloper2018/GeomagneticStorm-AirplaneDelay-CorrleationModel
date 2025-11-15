from pathlib import Path

folder = Path(r"C:\Users\arshp\OneDrive\Desktop\Terra Research")
print("Folder exists?", folder.exists())

# List all files with their repr() so hidden characters show up
for p in sorted(folder.iterdir()):
    print(repr(p.name))
