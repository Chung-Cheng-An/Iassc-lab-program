import re
import sys

usd_path = r"C:\Users\user\Desktop\IsaacLab\openarm_isaac_lab\source\openarm\openarm\tasks\manager_based\openarm_manipulation\usds\openarm_bimanual\openarm_bimanual.usd"

try:
    with open(usd_path, "rb") as f:
        content = f.read()
except Exception as e:
    print(f"Error reading file: {e}")
    sys.exit(1)

# Extract all ASCII strings of length >= 4
strings = re.findall(b"[a-zA-Z0-9_]{4,}", content)
strings = set([s.decode('ascii') for s in strings])

joints = sorted([s for s in strings if 'joint' in s.lower()])
links = sorted([s for s in strings if 'link' in s.lower() or 'hand' in s.lower() or 'ee' in s.lower()])

print("Joint names found:")
for j in joints:
    print("  ", j)

print("\nLink/Hand names found:")
for l in links:
    print("  ", l)
