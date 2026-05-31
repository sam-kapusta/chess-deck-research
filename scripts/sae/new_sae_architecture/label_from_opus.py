"""
Label features from Opus-profiled positions.
For each SAE, for each feature:
- Look at top-20 positions (from opus_profiles)
- Each position has an Opus analysis with tactical_motif, blunder_summary, best_moves_analysis
- Send batches to Sonnet to generate chip + description
"""
import json, os, boto3, time

BASE = "/home/ec2-user/SageMaker/chess-stage-a"
SAE_DIR = BASE + "/output/maia3_sae"
CLIENT = boto3.client("bedrock-runtime", region_name="us-east-1")
MODEL = "us.anthropic.claude-sonnet-4-6"
BATCH_SIZE = 12  # features per Sonnet call
JUNK = ["insufficient","unclassified","unavailable","unanalyzed","no data","no example"]

def call_sonnet(prompt, max_tokens=4000):
    resp = CLIENT.invoke_model(
        modelId=MODEL,
        body=json.dumps({"anthropic_version":"bedrock-2023-05-31",
                         "max_tokens":max_tokens,
                         "messages":[{"role":"user","content":prompt}]}))
    text = json.loads(resp["body"].read())["content"][0]["text"].strip()
    if text.startswith("```"):
        text = text.split("```")[1].lstrip("json").strip()
    return json.loads(text)

def format_feature_for_labeling(fi, fd):
    """Format one feature's top examples for the prompt."""
    examples = fd.get("examples", [])[:8]
    if not examples:
        return None
    lines = []
    for ex in examples:
        an = ex.get("analysis", {})
        if not isinstance(an, dict): continue
        motif   = an.get("tactical_motif", "")
        summary = an.get("blunder_summary", "")[:120]
        best    = an.get("best_moves_analysis", "")[:80]
        if summary:
            lines.append(f"  [{motif}] {summary}" + (f" | Best: {best}" if best else ""))
    if not lines:
        return None
    return f"Feature {fi} (fire_rate={fd['fire_rate']:.3f}):\n" + "\n".join(lines)

def label_sae(name, profiles_path, out_path):
    print(f"\n{'='*50}")
    print(f"Labeling {name}")
    print(f"{'='*50}")

    if not os.path.exists(profiles_path):
        print(f"  Profile not found: {profiles_path}")
        return {}

    profiles = json.load(open(profiles_path))
    
    # Load existing labels if any (resume support)
    existing = {}
    if os.path.exists(out_path):
        existing = {str(k):v for k,v in json.load(open(out_path)).items()}
        print(f"  Resuming: {len(existing)} already labeled")

    # Features that need labeling
    to_label = []
    for fi, fd in profiles.items():
        if fi in existing:
            v = existing[fi]
            chip = v.get("chip","").lower()
            if not any(j in chip for j in JUNK): continue  # already good
        if format_feature_for_labeling(fi, fd):
            to_label.append((fi, fd))

    print(f"  To label: {len(to_label)} features")
    labels = dict(existing)

    for batch_start in range(0, len(to_label), BATCH_SIZE):
        batch = to_label[batch_start:batch_start+BATCH_SIZE]
        
        feat_texts = []
        for fi, fd in batch:
            text = format_feature_for_labeling(fi, fd)
            if text: feat_texts.append(text)
        
        if not feat_texts: continue

        prompt = f"""You are a chess coach building a coaching taxonomy. Each SAE feature represents a recurring mistake pattern.

For each feature below, based on the example positions:
1. Write a SHORT CHIP (3-6 words, noun phrase, the coaching lesson a player needs)
2. Write a DESCRIPTION (2 sentences: what mistake, what should have been played)

The chip should sound like real coaching advice: "Bishop developed to attacked square", "Recaptured while queen hung", "Knight walked into fork", "King moved while under attack", etc.

Return JSON array: [{{"feature_id": "N", "chip": "...", "description": "..."}}]

{chr(10).join(feat_texts)}

Return ONLY the JSON array. No markdown."""

        try:
            results = call_sonnet(prompt)
            for r in results:
                fi = str(r.get("feature_id",""))
                if fi in [str(f) for f,_ in batch]:
                    labels[fi] = {"chip": r.get("chip",""), "description": r.get("description","")}
        except Exception as e:
            print(f"  batch {batch_start//BATCH_SIZE} error: {e}")

        done = min(batch_start+BATCH_SIZE, len(to_label))
        if done % (BATCH_SIZE*5) == 0 or done == len(to_label):
            # Save checkpoint
            json.dump(labels, open(out_path,"w"), indent=1)
            print(f"  {done}/{len(to_label)} labeled, saved", flush=True)

        time.sleep(0.3)  # rate limit

    json.dump(labels, open(out_path,"w"), indent=1)
    good = sum(1 for v in labels.values()
               if v.get("chip") and not any(j in v.get("chip","").lower() for j in JUNK))
    print(f"  Done: {len(labels)} total, {good} good labels ({good/max(len(labels),1)*100:.0f}%)")
    return labels

if __name__ == "__main__":
    for name in ["option_a", "board_diff", "l2l7"]:
        profiles_path = f"{SAE_DIR}/{name}_opus_profiles.json"
        out_path      = f"{SAE_DIR}/{name}_opus_labels.json"
        label_sae(name, profiles_path, out_path)
    print("\nAll done.")
