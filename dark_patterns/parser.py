import requests
import sys

if len(sys.argv) < 2:
    print("Usage: python parser.py <URL>")
    sys.exit(1)

url = sys.argv[1]

# --- GET request ---
res_get = requests.get("http://127.0.0.1:8000/predict/url", params={"url": url})
print("GET /predict/url response:")
try:
    print(res_get.json())
except Exception:
    print(res_get.text)

print("\n" + "=" * 80 + "\n")

# --- POST request ---
res_post = requests.post("http://127.0.0.1:8000/predict/url", json={"url": url})
print("POST /predict/url response:")
data = res_post.json()

if "analysis" in data:
    analysis = data["analysis"]

    if "ml_predictions" in analysis:
        ml_preds = analysis["ml_predictions"]["all_predictions"]
        print("\n--- Text Predictions ---")
        for pred in ml_preds[:10]:  # show first 10
            print(f"Snippet: {pred['text'][:80]}...")
            print(f"Category: {pred['category']}")
            print(f"Confidence: {pred['confidence']:.2f}")
            print("-" * 40)

    if "ui_patterns" in analysis:
        print("\n--- UI Patterns Detected ---")
        for ui in analysis["ui_patterns"]:
            print(f"- {ui['name']} ({ui['category']})")
