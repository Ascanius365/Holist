import os
import sys
import argparse
import pandas as pd
import json

sys.path.append(".")
from premem_module.src.utils.io import read_jsonl, save_pickle


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="minecraft")
    parser.add_argument("--model_name", type=str, default="qwen/qwen-2.5-coder-32b-instruct")
    parser.add_argument("--embedding_model_name", default="NovaSearch/stella_en_400M_v5", type=str)
    parser.add_argument("--data_dir", type=str, default="dataset/processed")
    parser.add_argument("--cache_dir", type=str, default=".cache")
    return parser.parse_args()


def fix_incomplete_json(json_str):
    # Entfernt Think-Blocks und korrigiert JSON-Struktur
    if "<think>" in json_str and "</think>" in json_str:
        json_str = json_str.split("</think>")[-1].strip()
    json_str = json_str.replace("```json", "").replace("```", "").strip()

    # Try 1: parse directly
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Try 2: find last } and try to parse
    try:
        last_brace = json_str.rfind("}")
        if last_brace != -1:
            return json.loads(json_str[:last_brace + 1])
    except json.JSONDecodeError:
        pass

    # Letzter Versuch: Gib leeres Dict zurück statt None
    # So wird die Session nicht komplett übersprungen
    print(f"⚠️  JSON nicht vollständig reparierbar, nutze leeres Dict")
    return {}


def main():
    args = parse_arguments()

    # Der Basis-Ordner für die Ergebnisse
    cache_base = args.cache_dir

    # WICHTIG: Sicherstellen, dass cache_base ein Verzeichnis ist
    if os.path.exists(cache_base) and not os.path.isdir(cache_base):
        print(f"Warnung: {cache_base} ist eine Datei, kein Ordner. Lösche Datei...")
        os.remove(cache_base)

    os.makedirs(cache_base, exist_ok=True)

    # Pfad zur Eingabedatei (Fakten aus der Extraktion)
    # Beachte: Der Pfad muss genau da liegen, wo run_extract_episodic_memory.py gespeichert hat
    input_file = f"{args.cache_dir}/ours.jsonl"

    if not os.path.exists(input_file):
        print(f"FEHLER: Eingabedatei nicht gefunden: {input_file}")
        return

    memories = read_jsonl(input_file)

    info_list = []
    for memory in memories:
        info = fix_incomplete_json(memory["response"])
        for key, value in info.items():
            if not isinstance(value, list): continue
            for v in value:
                if not v.get("value"): continue
                info_list.append({
                    "session_id": memory["session_id"],
                    "role": memory["role"],
                    "session_date": memory["session_date"],
                    "information_type": key,
                    "key": v.get("key", "info"),
                    "value": v["value"],
                    "date": v.get("date", "2024-01-01"),
                    "message_id": v.get("message_id", "m0")
                })

    if not info_list:
        print("Keine Informationen zum Embedden gefunden!")
        return

    info_df = pd.DataFrame(info_list)
    text_for_embed = (info_df["key"] + ": " + info_df["value"].astype(str)).tolist()

    # 2. Embeddings generieren
    from premem_module.src.db import EmbeddingDB
    import asyncio

    async def get_embedder():
        db = await EmbeddingDB.create(
            dataset_name=args.dataset_name,
            embedding_model_name=args.embedding_model_name,
            mode="turn",
            base_cache_dir=args.cache_dir,
            data_dir=args.data_dir,
        )
        return db.embedder

    embedder = asyncio.run(get_embedder())
    print(f"Erstelle Embeddings für {len(text_for_embed)} Fakten...")
    all_embeddings = embedder.create(text_for_embed)

    # 3. Speichern im Format, das run_reasoning.py erwartet
    # Wir nutzen 'global' als Key, da wir keine QA-IDs haben
    final_embeddings = {"global": all_embeddings}
    id2index = {"global": {i: i for i in range(len(info_list))}}
    id2text = {"global": {
        i: {"text": f"[{row['key']}]: {row['value']}", "session_date": row['session_date']}
        for i, row in info_df.iterrows()
    }}
    session_pool = {"global": info_df}

    save_pickle(final_embeddings, f"{cache_base}/embeddings.pkl")
    save_pickle(id2index, f"{cache_base}/id2index.pkl")
    save_pickle(id2text, f"{cache_base}/id2text.pkl")
    save_pickle(session_pool, f"{cache_base}/session_pool.pkl")

    print(f"ERFOLG: Dateien in {cache_base} gespeichert.")


if __name__ == "__main__":
    main()