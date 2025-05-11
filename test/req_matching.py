with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()
with open("requirements2.txt", "r") as f:
    requirements2 = f.read().splitlines()
    
    
requirements = [req.split("==")[0] for req in requirements]
requirements2 = [req.split("==")[0] for req in requirements2]

for req in requirements2:
    if req not in requirements:
        print(f"Missing requirement: {req}")
