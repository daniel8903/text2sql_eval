import requests

resp = requests.post(
    "http://localhost:11434/api/generate",
    json={"model": "llama2", "prompt": "Hi!", "stream": False}
)
data = resp.json()

# Extract stats
pe_dur = data["prompt_eval_duration"]  # in nanoseconds
ev_dur = data["eval_duration"]          # in nanoseconds

# Convert to seconds
pe_sec = pe_dur / 1e9
ev_sec = ev_dur / 1e9

# Compute tokens/sec
prompt_tps = pe_count / pe_sec if pe_sec else None
eval_tps = ev_count / ev_sec if ev_sec else None

print(f"Prompt tokens: {pe_count} in {pe_sec:.3f}s  →  {prompt_tps:.1f} TPS")
print(f"Output tokens: {ev_count} in {ev_sec:.3f}s  →  {eval_tps:.1f} TPS")
