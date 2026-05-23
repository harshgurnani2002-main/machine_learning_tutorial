import os
import re

files_to_fix = [
    "src/components/Simulators/GradientBoostingSimulator.tsx",
    "src/components/Simulators/KNNClassifierSimulator.tsx",
    "src/components/Simulators/NaiveBayesSimulator.tsx",
    "src/components/Simulators/PCASimulator.tsx",
    "src/components/Simulators/SVMSimulator.tsx",
    "src/components/Simulators/LogisticRegressionSimulator.tsx"
]

replacements = {
    "ctx.fillStyle = '#0f172a';": "const grad = ctx.createLinearGradient(0, 0, 0, h);\n    grad.addColorStop(0, '#FAF6EE');\n    grad.addColorStop(1, '#F4EFE6');\n    ctx.fillStyle = grad;",
    "ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';": "ctx.strokeStyle = 'rgba(110, 98, 87, 0.1)';",
    "ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';": "ctx.strokeStyle = 'rgba(110, 98, 87, 0.1)';",
    "'#f43f5e'": "'#B6532B'",
    "'#38bdf8'": "'#C18C3B'",
    "'#a78bfa'": "'#3B7A57'",
    "'rgba(244, 63, 94, 0.2)'": "'rgba(182, 83, 43, 0.15)'",
    "'rgba(56, 189, 248, 0.2)'": "'rgba(193, 140, 59, 0.15)'",
    "'rgba(167, 139, 250, 0.2)'": "'rgba(59, 122, 87, 0.15)'",
    "bg-slate-100": "bg-[#FAF6EE] border border-[#E5DDD0]",
    "bg-slate-50": "bg-[#FAF6EE]",
    "border-slate-200": "border-[#E5DDD0]",
    "border-slate-800": "border-[#E5DDD0]",
    "bg-slate-900": "bg-[#F4EFE6]",
    "bg-[#2E251E] border border-[#4A3D31]": "bg-[#F4EFE6] border border-[#E5DDD0]",
    "bg-white/10 backdrop-blur-md": "bg-[#2E251E]/80 backdrop-blur-md",
    "border-white/20": "border-[#E5DDD0]/30",
    "text-white shadow-lg pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity": "text-[#FAF6EE] shadow-lg pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity",
    "text-slate-900": "text-[#2E251E]",
    "text-slate-700": "text-[#2E251E]",
    "text-slate-500": "text-[#6E6257]",
    "text-slate-400": "text-[#6E6257]",
    "bg-slate-200": "bg-[#E5DDD0]",
    "hover:bg-slate-200": "hover:bg-[#F4EFE6]",
    "bg-rose-500": "bg-[#B6532B]",
    "bg-sky-500": "bg-[#C18C3B]",
    "text-rose-500": "text-[#B6532B]",
    "text-sky-500": "text-[#C18C3B]",
    "bg-red-50 text-red-600 hover:bg-red-100": "border border-[#E5DDD0] bg-[#FAF6EE] hover:bg-[#F4EFE6] text-[#6E6257] hover:text-[#B6532B]",
    "bg-emerald-50": "bg-[#3B7A57]/10",
    "text-emerald-600": "text-[#3B7A57]",
    "text-emerald-500": "text-[#3B7A57]",
    "accent-emerald-500": "accent-[#3B7A57]",
    "bg-emerald-600": "bg-[#3B7A57]",
    "bg-fuchsia-50": "bg-[#B6532B]/10",
    "text-fuchsia-600": "text-[#B6532B]",
    "text-fuchsia-500": "text-[#B6532B]",
    "accent-fuchsia-500": "accent-[#B6532B]",
    "bg-indigo-50": "bg-[#B6532B]/10",
    "text-indigo-600": "text-[#B6532B]",
    "text-indigo-500": "text-[#B6532B]",
    "accent-indigo-500": "accent-[#B6532B]",
    "bg-amber-50": "bg-[#C18C3B]/10",
    "text-amber-600": "text-[#C18C3B]",
    "text-amber-500": "text-[#C18C3B]",
    "accent-amber-500": "accent-[#C18C3B]",
    "bg-amber-500": "bg-[#C18C3B]",
    "text-orange-500": "text-[#B6532B]",
    "ctx.strokeStyle = '#ffffff';": "ctx.strokeStyle = '#FAF6EE';",
    "bg-white border border-[#E5DDD0]": "bg-[#FAF6EE] border border-[#E5DDD0]"
}

for file_path in files_to_fix:
    full_path = os.path.join("/home/legion/fastapi/machine_learning_tutorial", file_path)
    if os.path.exists(full_path):
        with open(full_path, "r") as f:
            content = f.read()
        
        for old, new in replacements.items():
            content = content.replace(old, new)
        
        # fix for the dark canvas background wrapper
        content = re.sub(r'bg-\[\#2E251E\].*?border-\[\#4A3D31\]', 'bg-[#F4EFE6] border border-[#E5DDD0]', content)
        
        with open(full_path, "w") as f:
            f.write(content)
        print(f"Fixed {file_path}")
    else:
        print(f"Not found: {file_path}")
