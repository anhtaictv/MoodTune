import os
import sys

# emotion_mlp.py/lexicon.py dùng import phẳng (from lexicon import ...), không
# phải package -> cần có backend/ trong sys.path để "import emotion_mlp" chạy
# được bất kể pytest được gọi từ đâu (repo root, backend/, ...).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
