"""Stable identities for the deduplicated relation snapshot."""
from copy import deepcopy
import hashlib
import json


def assign_relation_ids(relations):
    result = [deepcopy(relation) for relation in relations]
    used = set()
    for relation in result:
        payload = {key: relation.get(key, "") for key in ("head", "relation", "tail", "evidence", "skill")}
        digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:20]
        base = str(relation.get("relation_id") or "r-" + digest)
        identity = base
        duplicate = 1
        while identity in used:
            duplicate += 1
            identity = f"{base}-{duplicate}"
        relation["relation_id"] = identity
        used.add(identity)
    return result
