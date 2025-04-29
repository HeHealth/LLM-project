import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
import gzip
from sklearn.model_selection import train_test_split
from typing import List, Dict, Optional

# Configuration
DATA_DIR = "/home/yzhong43/redTeam"
OUTPUT_DIR = "/home/yzhong43/redTeam/processed_data"
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
RANDOM_SEED = 42
MIN_RATING_HARMFUL = 3
MAX_RATING_SAFE = 1

# Ensure output directory exists
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def load_jsonl_gz(file_path: str) -> List[Dict]:
    """Load JSONL.GZ file into a list of dictionaries (line-by-line)."""
    data = []
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        return []
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON at line {i} in {file_path}: {e}")
                except UnicodeDecodeError as e:
                    print(f"Warning: Unicode decode error at line {i} in {file_path}: {e}")
        print(f"Loaded {len(data)} items from {file_path}")
        return data
    except gzip.BadGzipFile:
        print(f"Error: {file_path} is not a valid gzip file")
        return []
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def load_json_array_gz(file_path: str) -> List[Dict]:
    """Load JSON.GZ file as a single JSON array."""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        return []
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            raw_data = json.load(f)
        if not isinstance(raw_data, list):
            print(f"Error: {file_path} is not a JSON array")
            return []
        print(f"Loaded {len(raw_data)} items from {file_path}")
        return raw_data
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {file_path}: {e}")
        return []
    except gzip.BadGzipFile:
        print(f"Error: {file_path} is not a valid gzip file")
        return []
    except UnicodeDecodeError:
        print(f"Warning: UTF-8 decode failed, trying latin-1 for {file_path}")
        try:
            with gzip.open(file_path, 'rt', encoding='latin-1') as f:
                raw_data = json.load(f)
            if not isinstance(raw_data, list):
                print(f"Error: {file_path} is not a JSON array")
                return []
            print(f"Loaded {len(raw_data)} items from {file_path} with latin-1")
            return raw_data
        except Exception as e:
            print(f"Error loading with latin-1: {e}")
            return []
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def extract_context(text: str) -> str:
    """Extract context (first Human: turn) from a conversation string."""
    lines = text.strip().split('\n')
    for line in lines:
        if line.startswith('Human:'):
            return line.replace('Human:', '').strip()
    return ''

def parse_preference_data(file_path: str, data_type: str) -> List[Dict]:
    """Parse human preference data into context, chosen, rejected format."""
    raw_data = load_jsonl_gz(file_path)
    parsed_data = []
    for item in raw_data:
        chosen = item.get('chosen', '')
        rejected = item.get('rejected', '')
        if not chosen or not rejected:
            print(f"Warning: Skipping item with missing chosen/rejected in {file_path}")
            continue
        context = extract_context(chosen) or extract_context(rejected)
        parsed_data.append({
            'context': context,
            'chosen': chosen,
            'rejected': rejected,
            'data_type': data_type
        })
    print(f"Parsed {len(parsed_data)} items from {file_path}")
    return parsed_data

def parse_red_team_data(file_path: str) -> List[Dict]:
    """Parse red-teaming data into context, chosen, rejected pairs."""
    raw_data = load_json_array_gz(file_path)
    parsed_data = []
    for item in raw_data:
        transcript = item.get('transcript', '')
        rating = item.get('rating', 0)
        min_harmlessness_score = item.get('min_harmlessness_score_transcript', 0.0)
        turns = transcript.strip().split('\n')
        human_turns = [t for t in turns if t.startswith('Human:')]
        assistant_turns = [t for t in turns if t.startswith('Assistant:')]
        if not human_turns or not assistant_turns:
            print(f"Warning: Skipping invalid transcript in {file_path}")
            continue
        context = human_turns[0].replace('Human:', '').strip()
        response = assistant_turns[0].replace('Assistant:', '').strip()
        if rating >= MIN_RATING_HARMFUL and min_harmlessness_score < 0:
            parsed_data.append({
                'context': context,
                'chosen': '',
                'rejected': response,
                'data_type': 'red_team',
                'rating': rating,
                'min_harmlessness_score': min_harmlessness_score
            })
        elif rating <= MAX_RATING_SAFE and min_harmlessness_score > 0:
            parsed_data.append({
                'context': context,
                'chosen': response,
                'rejected': '',
                'data_type': 'red_team',
                'rating': rating,
                'min_harmlessness_score': min_harmlessness_score
            })
    print(f"Parsed {len(parsed_data)} items from {file_path}")
    return parsed_data

def combine_and_split_data(preference_data: List[Dict], red_team_data: List[Dict]) -> Dict[str, List[Dict]]:
    """Combine preference and red-team data, then split into train/val/test."""
    combined_data = preference_data + red_team_data
    print(f"Combined {len(combined_data)} items (preference: {len(preference_data)}, red_team: {len(red_team_data)})")
    
    for item in combined_data:
        if item['data_type'] == 'red_team':
            if not item['chosen']:
                item['chosen'] = "I'm sorry, I can't assist with that request."
            if not item['rejected']:
                item['rejected'] = "This is a harmful response."
    
    np.random.seed(RANDOM_SEED)
    combined_df = pd.DataFrame(combined_data)
    train_data, temp_data = train_test_split(combined_df, test_size=(SPLIT_RATIOS['val'] + SPLIT_RATIOS['test']), random_state=RANDOM_SEED)
    val_data, test_data = train_test_split(temp_data, test_size=SPLIT_RATIOS['test']/(SPLIT_RATIOS['val'] + SPLIT_RATIOS['test']), random_state=RANDOM_SEED)
    
    splits = {
        'train': train_data.to_dict('records'),
        'val': val_data.to_dict('records'),
        'test': test_data.to_dict('records')
    }
    print(f"Split sizes: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    return splits

def extract_ppo_prompts(data: List[Dict]) -> List[Dict]:
    """Extract prompts for PPO training from contexts or chosen/rejected."""
    prompts = []
    seen_prompts = set()
    for item in data:
        context = item.get('context', '')
        if context and context not in seen_prompts:
            prompts.append({'prompt': context})
            seen_prompts.add(context)
        else:
            for field in ['chosen', 'rejected']:
                prompt = extract_context(item.get(field, ''))
                if prompt and prompt not in seen_prompts:
                    prompts.append({'prompt': prompt})
                    seen_prompts.add(prompt)
                    break
    print(f"Extracted {len(prompts)} unique PPO prompts")
    return prompts

def save_jsonl(data: List[Dict], file_path: str):
    """Save data to JSONL file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        print(f"Saved {len(data)} items to {file_path}")
    except Exception as e:
        print(f"Error saving {file_path}: {e}")

def main():
    """Main function to process datasets."""
    preference_files = [
        (os.path.join(DATA_DIR, 'helpful-base/train.jsonl.gz'), 'helpful_base'),
        (os.path.join(DATA_DIR, 'helpful-base/test.jsonl.gz'), 'helpful_base'),
        (os.path.join(DATA_DIR, 'helpful-online/train.jsonl.gz'), 'helpful_online'),
        (os.path.join(DATA_DIR, 'helpful-online/test.jsonl.gz'), 'helpful_online'),
        (os.path.join(DATA_DIR, 'helpful-rejection-sampled/train.jsonl.gz'), 'helpful_rejection'),
        (os.path.join(DATA_DIR, 'helpful-rejection-sampled/test.jsonl.gz'), 'helpful_rejection'),
        (os.path.join(DATA_DIR, 'harmless-base/train.jsonl.gz'), 'harmless_base'),
        (os.path.join(DATA_DIR, 'harmless-base/test.jsonl.gz'), 'harmless_base')
    ]
    
    preference_data = []
    for file_path, data_type in preference_files:
        if os.path.exists(file_path):
            preference_data.extend(parse_preference_data(file_path, data_type))
        else:
            print(f"Warning: Skipping missing file {file_path}")
    
    red_team_file = os.path.join(DATA_DIR, 'red-team-attempts/red_team_attempts.jsonl.gz')
    red_team_data = parse_red_team_data(red_team_file)
    
    splits = combine_and_split_data(preference_data, red_team_data)
    
    save_jsonl(splits['train'], os.path.join(OUTPUT_DIR, 'preference_train.jsonl'))
    save_jsonl(splits['val'], os.path.join(OUTPUT_DIR, 'preference_val.jsonl'))
    save_jsonl(splits['test'], os.path.join(OUTPUT_DIR, 'preference_test.jsonl'))
    
    ppo_prompts = extract_ppo_prompts(splits['train'])
    save_jsonl(ppo_prompts, os.path.join(OUTPUT_DIR, 'ppo_prompts.jsonl'))

if __name__ == "__main__":
    main()
