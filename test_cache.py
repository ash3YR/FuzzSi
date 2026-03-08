from semantic_cache import semantic_search

queries = [
    ("kidney stone treatment", "how to treat kidney stones"),
    ("car engine problem", "vehicle motor issue"),
    ("gun law debate", "firearm regulation discussion"),
]

thresholds = [0.75, 0.85, 0.95]

for t in thresholds:
    print(f"\n======================")
    print(f"THRESHOLD = {t}")
    print(f"======================")

    for q1, q2 in queries:
        print(f"\nQuery 1: {q1}")
        semantic_search(q1, threshold=t)

        print(f"Query 2: {q2}")
        semantic_search(q2, threshold=t)
