with open('src/data/modules/gradientBoosting.ts', 'r') as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    if 240 <= i <= 250:
        pass
    new_lines.append(line)
