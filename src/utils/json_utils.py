import json

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, file_path, lock=None):
    # Acquire lock
    if lock is not None:
        lock.acquire()

    # Write to file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    # Release lock
    if lock is not None:
        lock.release()
