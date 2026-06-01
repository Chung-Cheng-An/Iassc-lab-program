import os

search_dir = r"C:\Users\user\Desktop\IsaacLab\openarm_isaac_lab"

for root, _, files in os.walk(search_dir):
    for f in files:
        if f.endswith('.py'):
            filepath = os.path.join(root, f)
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    content = file.read()
                    if 'openarm_bimanual' in content or 'joint_names' in content or 'OPENARM' in content:
                        print(f"--- MATCH IN: {filepath} ---")
                        # print lines with joint
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if 'joint_names' in line or 'openarm' in line.lower() or 'left' in line.lower() or 'right' in line.lower():
                                print(f"  {i+1}: {line.strip()}")
            except:
                pass
