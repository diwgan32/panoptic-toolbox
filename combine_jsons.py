import json
import random

NUM_CHUNKS = 32

def where_id_exists(data, idx):
    l = []
    for i in range(len(data["images"])):
        if (data["images"]["id"] == idx):
            l.append(i)

    return l

def check_ids(data):
    for i in range(10):
        idx = random.randint(0, len(data["images"]))
        l = where_id_exists(data, idx)
        if (len(l) > 1):
            print(idx, l)


if __name__ == "__main__":
    data = {}
    for i in range(NUM_CHUNKS):
        f = open(f"panoptic_training_{i}.json", "r")
        file_data = json.load(f)
        f.close()
        if (i == 0):
            data = file_data
        else:
            data["images"] += file_data["images"]
            data["annotations"] += file_data["annotations"]


    f = open("panoptic_training.json", "w")
    where_id_exists(data)
    json.dump(data, f)
    f.close()