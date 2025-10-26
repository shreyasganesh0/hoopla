from lib.hybrid_search import rrf_search
import json

def evaluvate_scores(limit):

    test_cases = []
    with open("data/golden_dataset.json", "r") as f_d:
        data = json.load(f_d)
        test_cases = data["test_cases"]

    for t in test_cases:

        total_retrieved = rrf_search(t["query"], 60, limit, "", "")

        c = 0
        ret_title = []
        for ret in total_retrieved:
            ret_title.append(ret["title"])
            if ret["title"] in t["relevant_docs"]:
                c += 1
        
        if len(total_retrieved) == 0:
            precision = 0.0
        else:
            precision = c / len(total_retrieved)

        print(f"- Query: {t['query']}")
        
        print(f"    - Precision@{limit}: {precision:.4f}")
        
        print(f"    - Retrieved: {", ".join(ret_title)}")
        
        print(f"    - Relevant: {", ".join(t['relevant_docs'])}")
