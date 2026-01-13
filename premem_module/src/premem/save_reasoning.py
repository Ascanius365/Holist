import os
import json
import pickle
import pandas as pd


def fix_incomplete_json(json_str):
    """Parse JSON mit Fehlerbehandlung"""
    if "<think>" in json_str:
        json_str = json_str.split("</think>")[-1].strip()

    json_str = json_str.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        last_brace_pos = json_str.rfind("}")
        if last_brace_pos != -1:
            try:
                return json.loads(json_str[:last_brace_pos + 1])
            except:
                pass
        return None


def force_save():
    """Extrahiere die vollständigen LLM-Responses und speichere sie als Pickle"""

    reasoning_input = ".cache/ours_reasoning.jsonl"
    output_dir = ".cache/"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(reasoning_input):
        print(f"❌ Fehler: {reasoning_input} nicht gefunden!")
        return

    reasoning_units = []
    total_processed = 0
    total_failed = 0

    with open(reasoning_input, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                total_processed += 1

                # Die response IST die wichtige Zusammenfassung!
                response_text = data.get('response', '')

                if not response_text:
                    total_failed += 1
                    continue

                # Parse die Response um extended_insight zu extrahieren
                resp = fix_incomplete_json(response_text)

                if resp and "extended_insight" in resp:
                    # Speichere die gesamte Response + Metadaten
                    reasoning_units.append({
                        "content": response_text,  # ← Die vollständige Response!
                        "hash_key": data.get('hash_key', 'unknown'),
                        "cluster_id": data.get('cluster_id', 0),
                        "session_count": data.get('session_count', 1),
                        "insights_count": len(resp.get("extended_insight", [])),
                        "model": data.get('model', 'unknown'),
                        "usage_tokens": data.get('usage', {}).get('total_tokens', 0)
                    })
                else:
                    total_failed += 1
            except Exception as e:
                total_failed += 1
                print(f"⚠️  Fehler beim Verarbeiten einer Zeile: {e}")

    if not reasoning_units:
        print("❌ Keine gültigen Reasoning Units gefunden!")
        return

    # Erstelle DataFrame und speichere
    final_df = pd.DataFrame(reasoning_units)
    output_file = os.path.join(output_dir, "reasoning_units.pkl")

    with open(output_file, 'wb') as f:
        pickle.dump(final_df, f)

    print(f"✅ ERFOLG: {len(reasoning_units)} Reasoning Units extrahiert!")
    print(f"   Verarbeitet: {total_processed}, Fehlgeschlagen: {total_failed}")
    print(f"   Gespeichert in: {output_file}")

    print_memory(output_file)


def print_memory(file_path):
    """Zeige nur die Zusammenfassungen (extended_insight) der Cluster"""
    if not os.path.exists(file_path):
        print(f"❌ Datei nicht gefunden: {file_path}")
        return

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    print("\n" + "=" * 80)
    print("CLUSTER-ZUSAMMENFASSUNGEN (RAG Memory)")
    print("=" * 80)

    if hasattr(data, 'iterrows'):
        for idx, (_, row) in enumerate(data.iterrows(), 1):
            content = row.get('content', '')
            session_count = row.get('session_count', 1)

            # Parse die Response um extended_insight zu extrahieren
            resp = fix_incomplete_json(content)

            if resp and "extended_insight" in resp:
                print(f"\n📌 Cluster {idx} ({session_count} Sessions kombiniert):")
                for insight in resp["extended_insight"]:
                    key = insight.get('key', 'unknown')
                    value = insight.get('value', '')
                    inf_type = insight.get('inference_type', 'unknown')
                    print(f"   • [{inf_type}] {key}: {value}")

    print("\n" + "=" * 80)
    print(f"✅ Insgesamt {len(data)} Cluster verarbeitet")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    force_save()