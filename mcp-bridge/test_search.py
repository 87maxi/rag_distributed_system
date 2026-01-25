import sys
import os
# Add current dir to path so we can import stdio_server
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stdio_server import get_query_vector, qdrant, COLLECTION_NAME, generate_search_variations

def test_rag():
    query = "indexer"
    print(f"Testing query: {query}")
    
    # 1. Test Embeddings
    print("1. Testing Embedding Generation...")
    vec = get_query_vector(query)
    if not vec:
        print("FAIL: get_query_vector returned None")
        return
    print(f"SUCCESS: Vector generated (len={len(vec)})")

    # 2. Test Qdrant Search
    print("2. Testing Qdrant Search...")
    try:
        res = qdrant.search(collection_name=COLLECTION_NAME, query_vector=vec, limit=5)
        print(f"Qdrant Results found: {len(res)}")
        for r in res:
            print(f" - {r.id}: {r.payload.get('path', 'nopath')} (score: {r.score})")
    except Exception as e:
        print(f"FAIL: Qdrant search error: {e}")

    # 3. Test Variations
    print("3. Testing Variations...")
    vars = generate_search_variations(query)
    print(f"Variations: {vars}")

if __name__ == "__main__":
    test_rag()
