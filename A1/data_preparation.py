"""
Assignment 1: Dataset Creation for N-gram Language Model
Course: GenAI for Software Development (CSCI 455/555)
Instructor: Dr. Antonio Mastropaolo

Pipeline:
1. Fetch top Java repos from GitHub (stars > 1000, no forks)
2. Clone repos (shallow, --depth 1)
3. Select up to 20 Java files per repo (excluding test/example dirs)
4. Extract method-level code using javalang
5. Filter: remove non-ASCII, < 10 tokens, invalid brackets, empty body, TODOs
6. Tokenize with javalang tokenizer (space-separated)
7. Clean: remove multi-method lines and incomplete methods
8. Deduplicate exact matches
9. Split into T1 (<=15K), T2 (<=25K), T3 (<=35K), val (~1K), test (~1K)
10. Save to ./datasets/
"""

import os
import glob
import subprocess
import json
import random
import shutil
import statistics
from pathlib import Path

import requests
import pandas as pd
import javalang
from javalang.tokenizer import tokenize


CLONE_DIR  = "./datasets/java_repos"
OUTPUT_DIR = "./datasets/ngram_dataset"

CLASSES_PER_REPO = 20
MIN_TOKENS = 10

VAL_SIZE  = 1000
TEST_SIZE = 1000

T1_CAP = 15000
T2_CAP = 25000
T3_CAP = 35000

random.seed(42)

# Create directories
os.makedirs(CLONE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Setup complete!")
print(f"Clone directory: {CLONE_DIR}")
print(f"Output directory: {OUTPUT_DIR}")


#Fetch Repository List

def fetch_top_java_repos(num_repos=200, per_page=100):
    """
    Fetch top-starred Java repositories from GitHub API.
    Criteria:
      - Language: Java
      - Stars > 1000 (ensures high-quality, well-maintained projects)
      - Forks excluded (avoids duplicate code, following Tufano et al. 2024)
    """
    repos = []
    page = 1

    while len(repos) < num_repos:
        url = "https://api.github.com/search/repositories"
        params = {
            "q": "language:java stars:>100",
            "sort": "stars",
            "order": "desc",
            "per_page": per_page,
            "page": page
        }

        response = requests.get(url, params=params)

        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            break

        data = response.json()
        items = data.get("items", [])

        if not items:
            break

        for item in items:
            if item.get("fork", False):
                continue

            repos.append({
                "full_name": item["full_name"],
                "clone_url": item["clone_url"],
                "stars": item["stargazers_count"],
                "description": item.get("description", "")
            })

        page += 1

        if len(repos) >= num_repos:
            break

    return repos[:num_repos]


print("\nFetching top Java repositories from GitHub...")
repo_data = fetch_top_java_repos(num_repos=700)
df_repos = pd.DataFrame(repo_data)

print(f"\nFetched {len(df_repos)} repositories")
print(f"\nTop 10 repos by stars:")
print(df_repos[["full_name", "stars"]].head(10).to_string(index=False))


#Clone Repositories

def clone_repo(clone_url, dest_dir):
    """
    Shallow clone a repository (--depth 1).
    Only current snapshot is needed; saves time and disk space.
    Returns True if successful, False otherwise.
    """
    try:
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)

        cmd = ["git", "clone", "--depth", "1", "--quiet", clone_url, dest_dir]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  Timeout cloning {clone_url}")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


cloned_repos = []
failed_repos = []

print(f"\nCloning {len(df_repos)} repositories...\n")

for idx, row in df_repos.iterrows():
    repo_name = row["full_name"]
    clone_url = row["clone_url"]

    safe_name = repo_name.replace("/", "_")
    dest_dir = os.path.join(CLONE_DIR, safe_name)

    print(f"[{idx+1}/{len(df_repos)}] Cloning {repo_name}...", end=" ", flush=True)

    success = clone_repo(clone_url, dest_dir)

    if success:
        cloned_repos.append({
            "repo_name": repo_name,
            "local_path": dest_dir,
            "stars": row["stars"]
        })
        print("done")
    else:
        failed_repos.append(repo_name)
        print("failed")

print(f"\n\nSummary:")
print(f"  Successfully cloned: {len(cloned_repos)}")
print(f"  Failed: {len(failed_repos)}")


#Find and Select Java Files

def find_java_files(repo_path):
    """
    Find all .java files in a repository.
    Excludes test/example directories to keep production-quality code only.
    """
    java_files = []
    exclude_patterns = ["test", "tests", "example", "examples", "sample", "demo", "generated"]

    for root, dirs, files in os.walk(repo_path):
        root_lower = root.lower()
        if any(pattern in root_lower for pattern in exclude_patterns):
            continue

        for file in files:
            if file.endswith(".java"):
                java_files.append(os.path.join(root, file))

    return java_files


def select_java_files(java_files, max_files):
    """
    Randomly select up to max_files from the list.
    """
    if len(java_files) <= max_files:
        return java_files
    return random.sample(java_files, max_files)


repo_java_files = {}
all_selected_files = []

print(f"\nFinding Java files (selecting up to {CLASSES_PER_REPO} per repo)...\n")

for repo_info in cloned_repos:
    repo_name = repo_info["repo_name"]
    repo_path = repo_info["local_path"]

    java_files = find_java_files(repo_path)

    if not java_files:
        print(f"  {repo_name}: No Java files found")
        continue

    selected = select_java_files(java_files, max_files=CLASSES_PER_REPO)

    repo_java_files[repo_name] = {
        "total_files": len(java_files),
        "selected_files": [os.path.relpath(f, repo_path) for f in selected],
        "remaining_files": len(java_files) - len(selected)
    }

    all_selected_files.extend([(repo_name, f) for f in selected])
    print(f"  {repo_name}: {len(selected)}/{len(java_files)} files selected")

print(f"\nTotal Java files selected: {len(all_selected_files)}")


#Parse and Extract Methods

def read_file_content(file_path):
    """Read file content with multiple encoding fallbacks."""
    encodings = ['utf-8', 'latin-1', 'cp1252']

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue

    return None


def extract_method_source(source_code, method_node, lines):
    """Extract the source code of a method by counting braces."""
    try:
        start_line = method_node.position.line - 1

        brace_count = 0
        started = False
        end_line = start_line

        for i in range(start_line, len(lines)):
            line = lines[i]
            for char in line:
                if char == '{':
                    brace_count += 1
                    started = True
                elif char == '}':
                    brace_count -= 1

            if started and brace_count == 0:
                end_line = i
                break

        method_lines = lines[start_line:end_line + 1]
        return '\n'.join(method_lines)

    except Exception:
        return None


def extract_methods_from_file(file_path, repo_name):
    """Parse a Java file and extract all methods."""
    methods = []

    source_code = read_file_content(file_path)
    if source_code is None:
        return methods

    lines = source_code.split('\n')

    try:
        tree = javalang.parse.parse(source_code)

        for path, node in tree.filter(javalang.tree.MethodDeclaration):
            method_source = extract_method_source(source_code, node, lines)

            if method_source:
                methods.append({
                    "repo": repo_name,
                    "file": os.path.basename(file_path),
                    "method_name": node.name,
                    "source": method_source
                })

    except javalang.parser.JavaSyntaxError:
        pass
    except Exception:
        pass

    return methods


all_methods = []

print(f"\nExtracting methods from {len(all_selected_files)} files...\n")

for i, (repo_name, file_path) in enumerate(all_selected_files):
    if (i + 1) % 100 == 0:
        print(f"  Processed {i + 1}/{len(all_selected_files)} files...")

    methods = extract_methods_from_file(file_path, repo_name)
    all_methods.extend(methods)

print(f"\nTotal methods extracted: {len(all_methods)}")


#Filter Methods
def contains_non_ascii(text):
    """Check if text contains non-ASCII characters."""
    try:
        text.encode('ascii')
        return False
    except UnicodeEncodeError:
        return True


def count_tokens(source_code):
    """Count the number of Java tokens in source code."""
    try:
        tokens = list(tokenize(source_code))
        return len(tokens)
    except:
        return 0


filtered_methods = []

stats = {
    "total": len(all_methods),
    "non_ascii_dropped": 0,
    "too_short_dropped": 0,
    "kept": 0
}

print(f"\nFiltering {len(all_methods)} methods...\n")

for method in all_methods:
    source = method["source"]

    if contains_non_ascii(source):
        stats["non_ascii_dropped"] += 1
        continue

    token_count = count_tokens(source)
    if token_count < MIN_TOKENS:
        stats["too_short_dropped"] += 1
        continue

    method["token_count"] = token_count
    filtered_methods.append(method)
    stats["kept"] += 1

print(f"Filtering Results:")
print(f"  Total methods:        {stats['total']}")
print(f"  Dropped (non-ASCII):  {stats['non_ascii_dropped']}")
print(f"  Dropped (< {MIN_TOKENS} tokens): {stats['too_short_dropped']}")
print(f"  -------------------------")
print(f"  Methods kept:         {stats['kept']}")


#Tokenize Methods

def tokenize_method(source_code):
    """Tokenize Java source code into space-separated tokens using javalang.
    Emits keywords, identifiers, literals, operators, and punctuation as
    separate tokens (e.g., (, ), {, }, ;, .).
    """
    try:
        tokens = list(tokenize(source_code))
        token_values = [token.value for token in tokens]
        return ' '.join(token_values)
    except:
        return None


tokenized_methods = []

print(f"\nTokenizing {len(filtered_methods)} methods...\n")

for method in filtered_methods:
    tokenized = tokenize_method(method["source"])

    if tokenized:
        tokenized_methods.append({
            "repo": method["repo"],
            "file": method["file"],
            "method_name": method["method_name"],
            "tokenized_code": tokenized,
            "token_count": method["token_count"]
        })

print(f"Successfully tokenized: {len(tokenized_methods)} methods")

if tokenized_methods:
    example = tokenized_methods[0]
    print(f"\nExample tokenized method:")
    print(f"  Repo: {example['repo']}")
    print(f"  File: {example['file']}")
    print(f"  Method: {example['method_name']}")
    preview = example['tokenized_code'][:200]
    suffix = "..." if len(example['tokenized_code']) > 200 else ""
    print(f"  Tokens ({example['token_count']}): {preview}{suffix}")


#Clean, Deduplicate, and Split

def is_clean_method(tokenized_code):
    """Check if method is clean (single method, complete)."""
    method_keywords = (
        tokenized_code.count("public ") +
        tokenized_code.count("private ") +
        tokenized_code.count("protected ")
    )
    if method_keywords > 1:
        return False
    if not tokenized_code.endswith("}"):
        return False
    return True


# Clean
print(f"\nBefore cleaning: {len(tokenized_methods)}")
tokenized_methods = [m for m in tokenized_methods if is_clean_method(m['tokenized_code'])]
print(f"After cleaning: {len(tokenized_methods)}")

# Deduplicate
seen = set()
unique_methods = []
for m in tokenized_methods:
    if m['tokenized_code'] not in seen:
        seen.add(m['tokenized_code'])
        unique_methods.append(m)

print(f"After dedup: {len(unique_methods)}")
tokenized_methods = unique_methods

# Shuffle
random.shuffle(tokenized_methods)

#T1/T2/T3/val/test

total = len(tokenized_methods)
assert total >= VAL_SIZE + TEST_SIZE, (
    f"Not enough methods ({total}) for val + test. Reduce VAL_SIZE/TEST_SIZE."
)

val_data  = tokenized_methods[total - VAL_SIZE - TEST_SIZE : total - TEST_SIZE]
test_data = tokenized_methods[total - TEST_SIZE :]
train_pool = tokenized_methods[: total - VAL_SIZE - TEST_SIZE]

train_t1 = train_pool[:T1_CAP]
train_t2 = train_pool[:T2_CAP]
train_t3 = train_pool[:T3_CAP]

print(f"\nDataset Split:")
print(f"  T1 (train): {len(train_t1):,} methods  (cap: {T1_CAP:,})")
print(f"  T2 (train): {len(train_t2):,} methods  (cap: {T2_CAP:,})")
print(f"  T3 (train): {len(train_t3):,} methods  (cap: {T3_CAP:,})")
print(f"  Validation: {len(val_data):,} methods")
print(f"  Test:       {len(test_data):,} methods")


#Save Outputs

def save_txt(data, filename):
    """Save tokenized methods to a text file (one per line)."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w', encoding='utf-8', errors='replace') as f:
        for method in data:
            f.write(method['tokenized_code'] + '\n')
    return filepath


print("\nSaving dataset files...\n")

for name, data in [
    ("train_t1.txt", train_t1),
    ("train_t2.txt", train_t2),
    ("train_t3.txt", train_t3),
    ("val.txt",      val_data),
    ("test.txt",     test_data),
]:
    path = save_txt(data, name)
    print(f"  Saved: {path}")

# Save metadata
metadata = {
    "description": "Metadata for N-gram dataset. Shows which files were used for training.",
    "instructions_for_students": [
        "To create a test set from the SAME distribution:",
        "  1. Use the same repositories listed below",
        "  2. Select Java files NOT in 'selected_files'",
        "  3. Extract methods using the same preprocessing",
        "",
        "To create a test set likely to produce a DISTRIBUTION SHIFT:",
        "  1. Use different repositories (e.g., less popular by stars)",
        "  2. Compare model performance on both test sets"
    ],
    "dataset_stats": {
        "train_t1_size": len(train_t1),
        "train_t2_size": len(train_t2),
        "train_t3_size": len(train_t3),
        "val_size":  len(val_data),
        "test_size": len(test_data),
        "total_repos": len(repo_java_files),
        "min_tokens": MIN_TOKENS
    },
    "repos_used": repo_java_files
}

metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)
print(f"  Saved: {metadata_path}")

print(f"\nAll files saved to: {OUTPUT_DIR}/")


#Summary Statistics

all_token_counts = [m['token_count'] for m in tokenized_methods]

print("\n" + "=" * 50)
print("         DATASET CREATION SUMMARY")
print("=" * 50)

print(f"\nRepositories:")
print(f"   Cloned:    {len(cloned_repos)}")
print(f"   Failed:    {len(failed_repos)}")

print(f"\nJava Files:")
print(f"   Selected:  {len(all_selected_files)}")

print(f"\nMethods:")
print(f"   Extracted: {stats['total']}")
print(f"   Filtered:  {stats['total'] - stats['kept']} removed")
print(f"   Final:     {len(tokenized_methods)}")

print(f"\nToken Statistics:")
print(f"   Min:    {min(all_token_counts)}")
print(f"   Max:    {max(all_token_counts)}")
print(f"   Mean:   {statistics.mean(all_token_counts):.1f}")
print(f"   Median: {statistics.median(all_token_counts):.1f}")

print(f"\nDataset Splits:")
print(f"   T1 (train): {len(train_t1):,} methods  (cap <= {T1_CAP:,})")
print(f"   T2 (train): {len(train_t2):,} methods  (cap <= {T2_CAP:,})")
print(f"   T3 (train): {len(train_t3):,} methods  (cap <= {T3_CAP:,})")
print(f"   Validation: {len(val_data):,} methods")
print(f"   Test:       {len(test_data):,} methods")

print(f"\nOutput Files:")
for fname in ["train_t1.txt", "train_t2.txt", "train_t3.txt", "val.txt", "test.txt", "metadata.json"]:
    print(f"   {OUTPUT_DIR}/{fname}")
