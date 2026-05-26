import spacy
import requests
import numpy as np
import time

# Load local Swedish NLP
nlp = spacy.load("sv_core_news_lg")

KARP_API = "https://spraakbanken4.it.gu.se/karp/v7/query/lexin"
OLLAMA_API = "http://localhost:11434/api/embed"
MODEL = "bge-m3" # Ensure you have run 'ollama pull bge-m3'

def get_embedding(texts):
    """Robust Ollama call with error reporting."""
    payload = {"model": MODEL, "input": texts}
    try:
        r = requests.post(OLLAMA_API, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        if "embeddings" in data:
            return np.array(data["embeddings"])
        elif "embedding" in data: # Fallback for some Ollama versions
            return np.array([data["embedding"]])
        else:
            raise KeyError(f"Unexpected Ollama response: {data}")
    except Exception as e:
        print(f"!!! OLLAMA ERROR: {e}")
        # If bge-m3 fails, try to fall back to nomic or exit
        return None

def get_lexin_definitions(word, lemma):
    """Fetches definitions and strips them for better embedding comparison."""
    # Aggressively fetch both fruit and track candidates
    search_terms = {word.lower(), lemma.lower()}
    if word.lower() == "banan":
        search_terms.update(["banan", "bana"])
        
    meanings = []
    for term in search_terms:
        params = {"q": f"equals|languages.baseform|{term}", "size": 10}
        r = requests.get(KARP_API, params=params)
        if r.status_code == 200:
            for hit in r.json().get('hits', []):
                entry = hit.get('entry', {})
                swe_info = next((l for l in entry.get('languages', []) if l.get('lang') == 'swe'), {})
                pos = swe_info.get('partOfSpeech', 'unknown')
                definition = entry.get('sense', {}).get('definition', {}).get('text', '')
                
                if definition:
                    meanings.append({
                        "id": entry.get('sense', {}).get('senseid', 'unknown'),
                        "pos": pos,
                        "definition": definition,
                        "term": term
                    })
    # Remove duplicates
    return list({m['id']: m for m in meanings}.values())

def solve_task(sentence, char_index):
    doc = nlp(sentence)
    token = next((t for t in doc if t.idx <= char_index < (t.idx + len(t.text))), None)
    if not token: return print(f"Ingen token vid index {char_index}")

    print(f"\n--- TEST: '{sentence}' ---")
    print(f"Analys: '{token.text}' | POS: {token.pos_}")

    # 1. Fetch candidates
    candidates = get_lexin_definitions(token.text, token.lemma_)
    
    # 2. POS Filter (Noun only for banan)
    pos_map = {"NOUN": "nn", "VERB": "vb", "AUX": "vb", "ADJ": "jj"}
    target_pos = pos_map.get(token.pos_, "")
    candidates = [c for c in candidates if target_pos in c['pos']]

    if not candidates:
        return print("Hittade inga definitioner.")

    # 3. Create context-aware query
    # We use a 'Fill in the blank' style which BGE-M3 is very good at.
    before = sentence[:token.idx]
    after = sentence[token.idx + len(token.text):]
    context_str = f"{before}<{token.text.upper()}>{after}"
    
    query = f"Vad är den mest korrekta definitionen av ordet <{token.text}> i detta sammanhang: {context_str}"
    
    # 4. Get Embeddings
    all_texts = [query] + [f"Definition: {c['definition']}" for c in candidates]
    embeds = get_embedding(all_texts)
    
    if embeds is None: return

    # 5. Score using Cosine Similarity
    query_vec = embeds[0]
    doc_vecs = embeds[1:]
    scores = np.dot(doc_vecs, query_vec) / (np.linalg.norm(doc_vecs, axis=1) * np.linalg.norm(query_vec))
    
    results = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    
    for c, s in results:
        print(f"  [{s:.4f}] {c['id']} | {c['definition']}")

    print(f"VINNARE: {results[0][0]['definition']}")

if __name__ == "__main__":
    s_mix = "En banan låg på banan där bilarna körde."
    
    t_start = time.perf_counter()
    
    # Index 3 is the 'b' in the first 'banan'
    solve_task(s_mix, 3)
    
    # Index 16 is the 'b' in the second 'banan'
    solve_task(s_mix, 16)
    
    print(f"\nTotal tid: {time.perf_counter() - t_start:.2f}s")