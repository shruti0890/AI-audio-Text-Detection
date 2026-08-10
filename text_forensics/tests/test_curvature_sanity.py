"""Quick sanity test for Phase 1.2 — curvature signal."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from text_forensics.signals.curvature import get_curvature

# 3 human-written sentences (varied, personal, imperfect rhythm)
human_texts = [
    "Yesterday I went to the market and completely forgot to buy eggs. My wife reminded me three times but I still managed to walk out with nothing useful. At least I got a good coffee on the way back.",
    "The trail was muddier than I expected. My boots sank with every step, and by the second mile I had given up trying to keep them clean. I turned back early, which felt like failure but also relief.",
    "He called me at 7am which is frankly too early for anyone. I didn't answer but texted back later saying I'd seen it. That seemed to work fine.",
]

# 3 AI-sounding sentences (uniform, polished, buzzword-heavy)
ai_texts = [
    "In the realm of modern technology, it is important to note that artificial intelligence represents a pivotal and transformative force that is reshaping industries across the globe. This testament to human ingenuity underscores the importance of ethical considerations in the development of these powerful systems.",
    "Leveraging cutting-edge machine learning algorithms, organizations can unlock unprecedented value from their data ecosystems. It is crucial to delve into the multifaceted dimensions of this technological paradigm to fully appreciate its transformative potential.",
    "The seamless integration of artificial intelligence into business workflows has emerged as a cornerstone of modern enterprise strategy. By harnessing the power of data-driven insights, companies are positioned to navigate the complexities of an ever-evolving marketplace.",
]

print("=== Human texts ===")
for i, text in enumerate(human_texts, 1):
    score = get_curvature(text)
    print(f"  H{i}: {score:.4f}" if score is not None else f"  H{i}: None")

print("\n=== AI texts ===")
for i, text in enumerate(ai_texts, 1):
    score = get_curvature(text)
    print(f"  A{i}: {score:.4f}" if score is not None else f"  A{i}: None")

print("\n=== Edge cases ===")
short = "Hello world."
score = get_curvature(short)
print(f"  Short (<20 tokens): {score}")
