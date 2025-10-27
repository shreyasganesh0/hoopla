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
        
        if len(total_retrieved) == 0 or len(t["relevant_docs"]) == 0:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        else:
            precision = c / len(total_retrieved)
            recall = c / len(t["relevant_docs"])
            if (precision + recall) == 0.0:
                f1 = 0.0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)

        print(f"- Query: {t['query']}")
        
        print(f"    - Recall@{limit}: {recall:.4f}")

        print(f"    - Precision@{limit}: {precision:.4f}")

        print(f"    - F1 Score: {f1:.4f}")
        
        print(f"    - Retrieved: {", ".join(ret_title)}")
        
        print(f"    - Relevant: {", ".join(t['relevant_docs'])}")
