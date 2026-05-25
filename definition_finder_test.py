import requests
import numpy as np

# Verified API Endpoints
LEMMA_API = "https://json-tagger.com/api/v1/tag"
KARP_API = "https://spraakbanken4.it.gu.se/karp/v7/query/saldo"
OLLAMA_API = "http://localhost:11434/api/embed"

def get_swedish_lemma(sentence, word_to_find):
    """Identifies the dictionary base-form of a word in a sentence."""
    payload = {"text": sentence}
    print(f"\n[API REQUEST] POST {LEMMA_API} with payload: {payload}")
    try:
        r = requests.post(LEMMA_API, json=payload, timeout=5)
        print(f"[API RESPONSE] Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            for sentence_data in data.get('sentences', []):
                for token in sentence_data:
                    if token['word'] == word_to_find:
                        lemma = token['lemma']
                        print(f"[API DATA] Tagger lemma found: '{lemma}' for word '{word_to_find}'")
                        return lemma
        else:
            print(f"[API WARNING] Tagger returned status {r.status_code}: {r.text[:100]}")
    except Exception as e:
        print(f"[API ERROR] Connection to tagger failed: {e}")
    print(f"[API DATA] Using fallback lemma: '{word_to_find}'")
    return word_to_find # Fallback

def get_definitions_from_saldo(lemma):
    """Fetches real definitions from SALDO via Karp v7."""
    params = {
        "q": f"equals|baseform|{lemma}"
    }
    print(f"\n[API REQUEST] GET {KARP_API} with params: {params}")
    r = requests.get(KARP_API, params=params)
    print(f"[API RESPONSE] Status: {r.status_code}")
    results = []
    if r.status_code == 200:
        hits = r.json().get('hits', [])
        print(f"[API DATA] Found {len(hits)} raw hits in SALDO")
        for hit in hits:
            entry = hit.get('entry', {})
            sense_id = entry.get('senseID', '')
            primary = entry.get('primary', '')
            baseforms = entry.get('baseform', [])
            
            # Exact baseform matching to filter out multi-word idioms/phrases
            if lemma not in baseforms:
                print(f"  - Skipping '{sense_id}' (baseforms: {baseforms}) - does not exactly match '{lemma}'")
                continue
                
            if sense_id and primary:
                clean_sense = sense_id.split('..')[0].replace('_', ' ')
                clean_primary = primary.split('..')[0].replace('_', ' ')
                desc = f"{clean_sense} - primär betydelse: {clean_primary}"
                results.append(desc)
                print(f"  + Kept '{sense_id}' -> Formatted: '{desc}'")
    else:
        print(f"[API ERROR] Failed to fetch from Karp: {r.text[:200]}")
    return results

def get_embedding(text, task="search_document"):
    payload = {"model": "nomic-embed-text-v2-moe", "input": f"{task}: {text}"}
    print(f"[API REQUEST] POST {OLLAMA_API} with text: '{text[:60]}...' (task: {task})")
    r = requests.post(OLLAMA_API, json=payload)
    if r.status_code == 200:
        embedding = r.json()["embeddings"][0]
        print(f"[API RESPONSE] Success! Embedding vector dimension: {len(embedding)}")
        return np.array(embedding)
    else:
        print(f"[API ERROR] Ollama returned status {r.status_code}: {r.text[:100]}")
        raise RuntimeError(f"Ollama embedding failed: {r.text}")

def solve_task(sentence, word, char_index=None):
    # Find the exact character index if not provided
    if char_index is None:
        char_index = sentence.find(word)
        
    print(f"\n=================== Solving Task ===================")
    print(f"Sentence: '{sentence}'")
    print(f"Word to define: '{word}' at character index {char_index}")
    
    if char_index != -1:
        highlighted_sentence = sentence[:char_index] + f"**{word}**" + sentence[char_index+len(word):]
    else:
        highlighted_sentence = sentence
    print(f"Highlighted context: '{highlighted_sentence}'")

    # 1. Find the base word (e.g. 'banan' -> 'bana')
    lemma = get_swedish_lemma(sentence, word)
    
    # Generate potential lemma candidates (especially for definite forms ending in 'an')
    candidates = [lemma]
    if lemma != word:
        candidates.append(word)
    if lemma.endswith('an') and lemma[:-1] not in candidates:
        candidates.append(lemma[:-1])
    if word.endswith('an') and word[:-1] not in candidates:
        candidates.append(word[:-1])
    
    print(f"Lemma candidates to query: {candidates}")

    # 2. Get the menu of meanings from SALDO
    meanings = []
    for cand in candidates:
        meanings.extend(get_definitions_from_saldo(cand))
    
    # Remove duplicates from meanings if any
    meanings = list(dict.fromkeys(meanings))
    
    print(f"\nCandidates menu of meanings: {meanings}")
    
    if not meanings: return "No definition found."
    if len(meanings) == 1: return meanings[0]

    # 3. Disambiguate with Nomic
    print(f"\n[Disambiguation] Embedding query context...")
    query_vec = get_embedding(f"Betydelse av '{word}' i: {highlighted_sentence}", "search_query")
    
    best_score = -1
    best_def = ""
    print(f"[Disambiguation] Calculating cosine similarity:")
    for m in meanings:
        m_vec = get_embedding(m, "search_document")
        score = np.dot(query_vec, m_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(m_vec))
        print(f"  - Similarity score for '{m}': {score:.4f}")
        if score > best_score:
            best_score = score
            best_def = m
            
    print(f"Selected Best Definition: '{best_def}' (score: {best_score:.4f})")
    print(f"====================================================\n")
    return best_def

# TEST
if __name__ == "__main__":
    test_1 = "Bilen körde av banan."
    print(f"Mening: {test_1}\nDef: {solve_task(test_1, 'banan')}\n")
    
    test_2 = "Apan åt en banan."
    print(f"Mening: {test_2}\nDef: {solve_task(test_2, 'banan')}\n")