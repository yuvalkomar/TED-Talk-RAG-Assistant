import requests
import json
import sys
import time

# Default URL
BASE_URL = "https://tedtalkrag2.vercel.app"

def test_stats():
    url = f"{BASE_URL}/api/stats"
    print(f"Testing GET {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        required_keys = {"chunk_size", "overlap_ratio", "top_k"}
        if not required_keys.issubset(data.keys()):
            print(f"FAILED: Missing keys in stats response. Got: {list(data.keys())}")
            return False
            
        print("SUCCESS: Stats endpoint returned valid JSON.")
        print(json.dumps(data, indent=2))
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False

def ask_question(question, scenario_name):
    url = f"{BASE_URL}/api/prompt"
    print(f"\n--- Scenario: {scenario_name} ---")
    print(f"Question: {question}")
    
    try:
        start_time = time.time()
        response = requests.post(url, json={"question": question})
        response.raise_for_status()
        duration = time.time() - start_time
        
        data = response.json()
        
        # Basic Validation
        required_keys = {"response", "context", "Augmented_prompt"}
        if not required_keys.issubset(data.keys()):
            print(f"FAILED: Missing top-level keys. Got: {list(data.keys())}")
            return False

        print(f"Status: SUCCESS ({duration:.2f}s)")
        print(f"Response: {data['response']}")
        
        print("Retrieved Contexts:")
        seen_titles = set()
        for ctx in data.get("context", []):
            title = ctx.get("title", "Unknown")
            if title not in seen_titles:
                print(f" - {title} (Score: {ctx.get('score', 'N/A')})")
                seen_titles.add(title)
        
        return True
        
    except Exception as e:
        print(f"FAILED: {e}")
        return False

def run_test_scenarios():
    print("\n" + "="*50)
    print("RUNNING TEST SCENARIOS")
    print("="*50)
    
    scenarios = [
        (
            "1. Precise Fact Retrieval",
            "Find a TED talk that discusses overcoming fear or anxiety. Provide the title and speaker."
        ),
        (
            "2. Multi-Result Topic Listing",
            "Which TED talk focuses on education or learning? Return a list of exactly 3 talk titles."
        ),
        (
            "3. Key Idea Summary Extraction",
            "Find a TED talk where the speaker talks about technology improving people's lives. Provide the title and a short summary of the key idea."
        ),
        (
            "4. Recommendation with Evidence-Based Justification",
            "I'm looking for a TED talk about climate change and what individuals can do in their daily lives. Which talk would you recommend?"
        )
    ]
    
    results = []
    for name, question in scenarios:
        success = ask_question(question, name)
        results.append(success)
        
    return all(results)

if __name__ == "__main__":
    print(f"Targeting: {BASE_URL}")
    
    # 1. Test Stats
    if not test_stats():
        sys.exit(1)
        
    # 2. Run Scenarios
    if run_test_scenarios():
        print("\n" + "="*50)
        print("ALL SCENARIOS PASSED (Technical Check)")
        print("="*50)
        sys.exit(0)
    else:
        print("\nSOME TESTS FAILED")
        sys.exit(1)
