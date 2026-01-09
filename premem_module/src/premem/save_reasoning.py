import os
import json
import pickle
import pandas as pd


def fix_incomplete_json(json_str):
    # 1. Entferne den <think> Block, falls vorhanden
    if "<think>" in json_str:
        json_str = json_str.split("</think>")[-1].strip()

    # 2. Markdown-Code-Blöcke entfernen
    json_str = json_str.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Reparatur-Logik für unvollständige Klammern
        last_brace_pos = json_str.rfind("}")
        if last_brace_pos != -1:
            try:
                return json.loads(json_str[:last_brace_pos + 1])
            except:
                # Spezifischer Fallback für PREMem-Listenstruktur
                return json.loads(json_str[:last_brace_pos] + "}\n  ]\n}")
        return None


def force_save():

    dataset = "minecraft"
    model = "openrouter/qwen/qwen-2.5-coder-32b-instruct"
    reasoning_input = f".cache/ours_reasoning.jsonl"  # Pfad prüfen!
    output_dir = f".cache/"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(reasoning_input):
        print(f"Fehler: {reasoning_input} nicht gefunden!")
        return

    reasoning_units = []
    with open(reasoning_input, 'r') as f:
        for line in f:
            data = json.loads(line)
            # Hier nutzen wir die Fix-Funktion
            resp = fix_incomplete_json(data['response'])

            if resp and "extended_insight" in resp:
                for insights in resp["extended_insight"]:
                    reasoning_units.append({
                        "content": f"[{insights.get('key')}]: {insights.get('value')}",
                        "source_sessions": data.get("session_id", "mc_all"),
                        "date": insights.get("date")
                    })

    if not reasoning_units:
        print("Keine gültigen Reasoning Units gefunden. Prüfe die JSON-Struktur in der Input-Datei.")
        return

    final_df = pd.DataFrame(reasoning_units)
    output_file = os.path.join(output_dir, "reasoning_units.pkl")

    with open(output_file, 'wb') as f:
        pickle.dump(final_df, f)

    print(f"ERFOLG: {len(reasoning_units)} Reasoning Units manuell gespeichert!")

    print_memory(".cache/reasoning_units.pkl")


def print_memory(file_path):
    if not os.path.exists(file_path):
        print(f"Datei nicht gefunden: {file_path}")
        return

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    seen_summaries = set()

    print("=" * 60)
    print("ZUSAMMENGEFASSTES WISSEN (Optimierte Anzeige)")
    print("=" * 60)

    # Fall 1: Daten sind ein Pandas DataFrame (wie im Force-Save-Skript)
    if hasattr(data, 'iterrows'):
        for _, row in data.iterrows():
            text = row.get('content', row.get('text', ''))
            if text and text not in seen_summaries:
                print(f"• {text}")
                seen_summaries.add(text)

    # Fall 2: Daten sind das originale PREMem-Dictionary Format (id2text)
    elif isinstance(data, dict):
        for sub_dict in data.values():
            if isinstance(sub_dict, dict):
                # Prüfe ob es ein verschachteltes Dict ist (id2text Struktur)
                for sub_content in sub_dict.values():
                    if isinstance(sub_content, dict):
                        text = sub_content.get('text', '')
                        if text and text not in seen_summaries:
                            print(f"• {text}")
                            seen_summaries.add(text)
            else:
                # Einfaches Dictionary Mapping
                text = sub_dict.get('text', '') if isinstance(sub_dict, dict) else str(sub_dict)
                if text not in seen_summaries:
                    print(f"• {text}")
                    seen_summaries.add(text)

    # Fall 3: Daten sind ein Numpy Array oder eine Liste
    elif hasattr(data, '__iter__'):
        for item in data:
            text = str(item)
            if text not in seen_summaries:
                print(f"• {text}")
                seen_summaries.add(text)

    print("\n" + "=" * 60)
    print(f"Das System hat insgesamt {len(seen_summaries)} Einträge geladen.")


if __name__ == "__main__":
    force_save()