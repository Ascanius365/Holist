from simple_chalk import chalk
import asyncio
import subprocess
import pickle
import json
import os
from datetime import datetime
from premem_module.src.db import EmbeddingDB


# Enhanced mock agent for the PreMem utility functions
class SimpleTokenCounter:
    def count_tokens(self, text):
        return len(str(text)) // 4

    def truncate_text(self, text, max_tokens):
        limit = max_tokens * 4
        if len(str(text)) > limit:
            return str(text)[:limit] + "..."
        return str(text)


db_instance = None
token_counter = SimpleTokenCounter()


def update_long_term_memory():
    """Start the PREMem pipeline with the appropriate arguments."""

    def run_pipeline():

        venv_python = "/home/benito/PycharmProjects/Holist/.venv/bin/python3"
        project_root = "/home/benito/PycharmProjects/Holist"

        pipeline_steps = [
            ("premem_module/src/premem/run_extract_episodic_memory.py", True, True, True),
            ("premem_module/src/premem/save_episodic_embedding.py", False, True, True),
            ("premem_module/src/premem/run_reasoning.py", True, True, True),
            ("premem_module/src/premem/save_reasoning.py", False, True, True)
        ]

        print(chalk.yellow("🧠 [PREMem] Start long-term memory update..."))

        for script_path, use_api, use_model, use_dataset in pipeline_steps:

            cmd = [venv_python, script_path]

            if use_dataset:
                cmd += ["--dataset_name", "minecraft"]
            if use_model:
                cmd += ["--model_name", "openrouter/qwen/qwen-2.5-coder-32b-instruct"]
            if use_api:
                cmd += ["--api_base", "https://openrouter.ai/api/v1"]
            if "run_reasoning" in script_path:
                cmd += ["--batch_size", "1"]
            if "save_reasoning" in script_path:
                cmd += ["--mode", "ours"]

            print(chalk.cyan(f"🚀 Execute: {script_path}"))

            try:
                result = subprocess.run(
                    cmd,
                    stdout=None,
                    stderr=None,
                    text=True,
                    cwd=project_root
                )

                if result.returncode == 0:
                    print(chalk.green(f"✅ {script_path} successfully completed."))
                else:
                    print(chalk.red(f"❌ Error in {script_path}:"))
                    print(chalk.gray(result.stderr))
                    return

            except Exception as e:
                print(chalk.red(f"❌ System error at {script_path}: {e}"))
                return

        print(chalk.green("✨ [PREMem] Update complete! The new knowledge is now ready."))

    fixed_conversion_v3()
    # Move to a background thread to prevent Minecraft from lagging
    import threading
    threading.Thread(target=run_pipeline, daemon=True).start()


def start_rag(observation_text, observation):
    global db_instance
    _ensure_memory_loaded()

    try:
        # Search for relevant entries
        retrieved_results = db_instance.retrieve(
            observation_text,
            k=10
        )

        print(f"🔍 RAG: {len(retrieved_results)} results retrieved")

        # Show some examples
        if retrieved_results:
            print(f"DEBUG - Erste 3 Ergebnisse:")
            for i, res in enumerate(retrieved_results[:3]):
                print(f"  [{i}] Text: {res.get('text', '')[:80]}...")
                print(f"      Score: {res.get('score')}")

        knowledge_entries = []
        seen = set()

        for res in retrieved_results:
            if not isinstance(res, dict):
                continue

            unit = res.get('text', '').strip()
            date = res.get('date', res.get('session_date', 'unknown'))

            if not unit:
                continue

            if unit.startswith("[user]: "):
                unit = unit[8:]

            if unit.startswith("[") and "]: " in unit:
                if unit not in seen:
                    knowledge_entries.append({
                        'text': unit,
                        'date': date,
                        'score': res.get('score', 0)
                    })
                    seen.add(unit)
                    print(f"✅ Added (Score: {res.get('score', 0):.3f}, Date: {date}): {unit[:70]}...")
            else:
                print(f"⭐️ Skipped: {unit[:60]}... (The format is incorrect.)")

        # Limit to top n best matches
        final_entries = knowledge_entries[:10]

        if not final_entries:
            print("⚠️ RAG: No valid reasoning units found.")
            return ""

        # Format for prompt - WITH RELATIVE DATE
        formatted_context = "\n".join([
            f"• [{format_relative_time(entry['date'], observation)}] {entry['text']}"
            for entry in final_entries
        ])

        enhanced_prompt = (
            f"### Abstract ideas as feedback:\n"
            f"{formatted_context}"
        )

        print(f"\n✅ RAG-prompt successfully generated ({len(final_entries)} Units)")
        print(enhanced_prompt)
        return enhanced_prompt

    except AttributeError as e:
        if "'dict' object has no attribute 'norm'" in str(e):
            print("⚠️ RAG: EmbeddingDB has an embedding error (dictionary instead of tensor). "
                  "Try manual fallback.")
            return _fallback_rag(observation_text, observation)
        else:
            print(f"❌ RAG Error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return ""
    except Exception as e:
        print(f"❌ RAG Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return ""


def _fallback_rag(observation_text, observation):

    try:
        import pickle
        import os
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        reasoning_pkl = ".cache/reasoning_units.pkl"

        if not os.path.exists(reasoning_pkl):
            print(f"⚠️ {reasoning_pkl} not found!")
            return ""

        # Lade Reasoning Units
        with open(reasoning_pkl, 'rb') as f:
            reasoning_df = pickle.load(f)

        if reasoning_df.empty:
            print("⚠️ Reasoning units are empty")
            return ""

        # Extrahiere Texte UND Daten
        texts = reasoning_df['content'].tolist() if 'content' in reasoning_df.columns else []
        texts = [str(t) for t in texts if t]

        # Extrahiere Daten
        dates = reasoning_df['date'].tolist() if 'date' in reasoning_df.columns else ['unknown'] * len(texts)
        dates = [str(d) for d in dates]

        if not texts:
            print("⚠️ No text in reasoning units found")
            return ""

        print(f"📊 Fallback-RAG: Use {len(texts)} reasoning units")

        # TF-IDF Similarity
        vectorizer = TfidfVectorizer(max_features=500, lowercase=True, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)
        query_vector = vectorizer.transform([observation_text])

        similarities = cosine_similarity(query_vector, tfidf_matrix)[0]
        top_k_indices = similarities.argsort()[-10:][::-1]

        knowledge_entries = []
        for idx in top_k_indices:
            if similarities[idx] > 0.0:  # Mindest-Threshold
                text = texts[idx]
                date = dates[idx] if idx < len(dates) else "unknown"

                if text.startswith("[") and "]: " in text:
                    knowledge_entries.append({
                        'text': text,
                        'date': date,
                        'score': similarities[idx]
                    })
                    print(f"✅ Fallback Added (Score: {similarities[idx]:.3f}, Date: {date}): {text[:70]}...")

        if not knowledge_entries:
            print("⚠️ Fallback-RAG: Keine validen Ergebnisse")
            return ""

        # Format for prompt - WITH RELATIVE DATE
        formatted_context = "\n".join([
            f"• [{format_relative_time(entry['date'], observation)}] {entry['text']}"
            for entry in knowledge_entries
        ])

        enhanced_prompt = (
            f"### Abstract ideas as feedback:\n"
            f"{formatted_context}"
        )

        print(f"\n✅ RAG-prompt generated successfully ({len(enhanced_prompt)} units)")
        print(enhanced_prompt)
        return enhanced_prompt

    except Exception as e:
        print(f"❌ Fallback-RAG error: {type(e).__name__}: {e}")
        return ""


def generate_rag_query(observation, memory_messages, history_text):
    """
    Creates a search query for the RAG system based on the current observation,
    the long-term summary, and the complete short-term memory.
    """
    # 1. Current situation
    current_obs = observation.get("observation", "")
    current_task = observation.get("status", {}).get("task", "")

    # 2. All entries from memory.messages
    past_interactions = ""
    for msg in memory_messages:
        past_interactions += f" Situation: {msg.get('input', '')} Action: {msg.get('output', '')}"

    full_query = (
        f"Task: {current_task} | "
        f"Current: {current_obs} | "
        f"History: {history_text} | "
        f"Past Interactions: {past_interactions}"
    )

    return full_query[-1200:].strip()


def _ensure_memory_loaded():
    """Loads the Reasoning Units directly from the PKL file into the EmbeddingDB."""
    global db_instance

    if db_instance is None:
        print(f"🧠 Initialise long term memory (Reasoning Units)...")

        reasoning_pkl = ".cache/reasoning_units.pkl"

        # Load PKL-File
        if not os.path.exists(reasoning_pkl):
            print(f"⚠️ {reasoning_pkl} not found!")
            return

        try:
            with open(reasoning_pkl, 'rb') as f:
                reasoning_df = pickle.load(f)
            print(f"✅ Reasoning Units geladen: {len(reasoning_df)} Einträge")

            # Convert DataFrame to sessions.pkl format
            session_data_list = []

            if hasattr(reasoning_df, 'iterrows'):
                for i, (_, row) in enumerate(reasoning_df.iterrows()):
                    unit_text = row.get('content', '')
                    if unit_text and unit_text.strip():
                        session_dict = {
                            "session_id": f"reasoning_{i}",
                            "session_date": row.get('date', "2024-01-01 Monday 12:00:00"),
                            "conversation": [{
                                "role": "user",
                                "content": unit_text,
                                "message_id": f"m_{i}"
                            }],
                            "others": {}
                        }
                        session_data_list.append(session_dict)

            temp_sessions_pkl = "premem_module/dataset/sessions.pkl"
            os.makedirs(os.path.dirname(temp_sessions_pkl), exist_ok=True)

            with open(temp_sessions_pkl, 'wb') as f:
                pickle.dump(session_data_list, f)

            print(f"✅ {len(session_data_list)} Sessions in {temp_sessions_pkl} gespeichert")

            # Load in EmbeddingDB
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Clear the old cache so that fresh embeddings can be created.
            import shutil
            cache_dirs = [
                ".cache/embeddings",
                ".cache/ours_embeddings"
            ]
            for cache_dir in cache_dirs:
                if os.path.exists(cache_dir):
                    try:
                        shutil.rmtree(cache_dir)
                        print(f"🧹 Cache deleted: {cache_dir}")
                    except:
                        pass

            db_instance = loop.run_until_complete(
                EmbeddingDB.create(
                    dataset_name="minecraft",
                    embedding_model_name="NovaSearch/stella_en_400M_v5",
                    mode="ours",
                    data_dir="premem_module/dataset/processed",
                    base_cache_dir=".cache"
                )
            )
            print("✅ EmbeddingDB successfully loaded with Reasoning Units!")

        except Exception as e:
            print(f"❌ Error loading Reasoning Units: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()


def fixed_conversion_v3():
    """Converts sessions.jsonl to sessions.pkl."""
    target_dir = "premem_module/dataset"
    input_file = os.path.join(target_dir, "sessions.jsonl")
    output_file = os.path.join(target_dir, "sessions.pkl")

    if not os.path.exists(input_file):
        print(f"⚠️ {input_file} nicht gefunden")
        return

    session_data_list = []
    failed_lines = 0

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"⚠️ Invalid JSON in line {i + 1}: {str(e)[:80]}")
                    print(f"   Line: {line[:100]}...")
                    failed_lines += 1
                    continue
                except Exception as e:
                    print(f"❌ Error reading line {i + 1}: {e}")
                    failed_lines += 1
                    continue

                if not isinstance(data, dict):
                    print(f"⚠️ Line {i + 1} is no JSON-object (skip)")
                    failed_lines += 1
                    continue

                session_dict = {
                    "session_id": data.get('session_id', f"mc_{i}"),
                    "session_date": data.get('date', data.get('session_date', "2024-01-01 Monday 12:00:00")),
                    "conversation": [{
                        "role": "user",
                        "content": data.get('text', ''),
                        "message_id": f"m_{i}"
                    }],
                    "others": {}
                }
                session_data_list.append(session_dict)

    except Exception as read_error:
        print(f"❌ Fehler beim Lesen von {input_file}: {read_error}")
        return

    os.makedirs(target_dir, exist_ok=True)

    try:
        with open(output_file, 'wb') as f:
            pickle.dump(session_data_list, f)
        print(f"✅ SUCCESS! {output_file} wurde aktualisiert.")
        print(f"📊 Gültige Sessions: {len(session_data_list)}")
        if failed_lines > 0:
            print(f"⚠️ Übersprungene Zeilen: {failed_lines}")
    except Exception as write_error:
        print(f"❌ Fehler beim Schreiben von {output_file}: {write_error}")


def format_relative_time(date_str, observation):
    """
    Converts a timestamp or time range into a relative natural language description.
    Handles formats like:
    - "2026-01-07 Wednesday 16:58:43"
    - "2026-01-07 Wednesday 15:13:49 to 2026-01-07 Wednesday 16:49:41"
    """
    try:
        # Reference time for calculation
        now = observation.get("time", "")

        raw_now = observation.get("time", "").replace("   - Current Time: ", "").strip()

        def parse_dt(s):
            s = s.strip()
            formats = [
                "%Y-%m-%d %A %H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %A %H:%M",
                "%Y-%m-%d"
            ]
            for fmt in formats:
                try:
                    return datetime.strptime(s, fmt)
                except ValueError:
                    continue
            return datetime.fromisoformat(s.split(' ')[0])

        now = parse_dt(raw_now)

        duration_str = ""
        if " to " in date_str:
            start_str, end_str = date_str.split(" to ")
            dt_start = parse_dt(start_str)
            dt_end = parse_dt(end_str)

            # Calculate duration
            duration = dt_end - dt_start
            d_hours = duration.seconds // 3600
            d_mins = (duration.seconds % 3600) // 60

            if duration.days > 0:
                duration_str = f" for {duration.days} days"
            elif d_hours > 0:
                duration_str = f" for {d_hours} hours"
            elif d_mins > 0:
                duration_str = f" for {d_mins} minutes"

            dt = dt_start  # Use start time for relative calculation
        else:
            dt = parse_dt(date_str)

        diff = now - dt
        days = diff.days
        seconds = diff.seconds
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60

        time_str = dt.strftime("%H:%M:%S")

        if days == 0:
            if hours == 0:
                if minutes == 0:
                    return f"Just now{duration_str} at {time_str}"
                return f"{minutes} minutes ago{duration_str} at {time_str}"
            return f"Today, {hours} hours ago{duration_str} at {time_str}"
        elif days == 1:
            return f"Yesterday{duration_str} at {time_str}"
        elif days < 7:
            return f"{days} days ago{duration_str} at {time_str}"
        else:
            weeks = days // 7
            return f"{weeks} weeks ago{duration_str} at {time_str}"

    except Exception:
        return date_str