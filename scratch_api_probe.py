"""Quick probe: print all Lexin entries for our test words."""
import requests, json

KARP_API = "https://spraakbanken4.it.gu.se/karp/v7/query/lexin"

def probe(term):
    params = {"q": f"equals|languages.baseform|{term}", "size": 20}
    r = requests.get(KARP_API, params=params)
    hits = r.json().get("hits", [])
    print(f"\n=== '{term}' ({len(hits)} hits) ===")
    for hit in hits:
        entry = hit["entry"]
        swe = next((l for l in entry.get("languages", []) if l.get("lang") == "swe"), {})
        pos = swe.get("partOfSpeech", "?")
        sense_id = entry.get("sense", {}).get("senseid", "?")
        defn = entry.get("sense", {}).get("definition", {}).get("text", "(no definition)")
        inflections = entry.get("inflectionTable", [])
        forms = [i.get("writtenForm") for i in inflections[:6]]
        print(f"  [{sense_id}] POS={pos}")
        print(f"    DEF: {defn}")
        print(f"    FORMS (first 6): {forms}")

for word in ["löpa"]:
    probe(word)
