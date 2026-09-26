"""Whitespace-robust evidence search and immutable source records.

The index ranks passages, not legal correctness. Explicit source scope is optional;
no jurisdiction is inferred from a field name or its language.
"""
import hashlib
import re
import unicodedata
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix

SCOPE_PREFIXES = {
    '기관 내부 규정에서': (1,),
    '기관 내부 개인정보보호 시행세칙에서': (1,),
    '대한민국 개인정보 보호법에서': (2,),
    'GDPR의 본문 또는 전문에서': (3,),
}

def evidence_record(doc):
    text = str(doc.get('text', doc.get('content', '')) or '')
    source = str(doc.get('source', 'Unknown'))
    return {'chunk_id': hashlib.sha256((source+'\n'+text).encode()).hexdigest()[:16],
            'source': source, 'text': text,
            'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'priority_level': int(doc.get('priority_level', 3)),
            'status': 'retrieved_context_not_adjudicated'}

def query_terms(query):
    for prefix in SCOPE_PREFIXES:
        if query.startswith(prefix): query = query[len(prefix):].strip(); break
    # Ignore the fixed English request boilerplate; retain actual field description.
    match = re.search(r"with the description '(.*?)' constitute", query, re.S)
    if match and match.group(1).strip(): return match.group(1)
    match = re.search(r'Column Description:\s*([^\n]+)', query)
    if match and match.group(1).strip(): return match.group(1)
    return query

def tokens(text):
    text = re.sub(r'\s+', '', unicodedata.normalize('NFKC', text).lower()).replace('##', '')
    return Counter(text[i:i+n] for n in (2, 3) for i in range(len(text)-n+1)
                   if re.fullmatch(r'[가-힣a-z0-9]+', text[i:i+n]))

class LegalLexicalIndex:
    def __init__(self, documents):
        self.documents = documents
        bags = [tokens(str(d.get('text', '') or '')) for d in documents]
        vocabulary = {}
        for bag in bags:
            for term in bag:
                if term not in vocabulary: vocabulary[term] = len(vocabulary)
        self.vocabulary = vocabulary
        row, col, val = [], [], []
        for i, bag in enumerate(bags):
            for term, count in bag.items():row.append(i); col.append(vocabulary[term]); val.append(count)
        matrix = csr_matrix((val, (row, col)), shape=(len(documents), len(vocabulary)), dtype=float)
        lengths = np.asarray(matrix.sum(axis=1)).ravel()
        df = np.asarray((matrix > 0).sum(axis=0)).ravel()
        average = float(lengths.mean()) if len(lengths) else 1.0
        idf = np.log(1+(len(documents)-df+.5)/(df+.5))
        matrix.data = matrix.data*2.2/(matrix.data+1.2*(.25+.75*lengths[matrix.nonzero()[0]]/(average or 1)))
        self.matrix = matrix.multiply(idf).tocsr()

    def rank(self, query, similarities, top_k, allowed_levels=None, gap=0.0):
        if top_k <= 0: return []
        if allowed_levels is None:
            allowed_levels = next((v for p,v in SCOPE_PREFIXES.items() if query.startswith(p)), None)
        if allowed_levels is not None:
            if not isinstance(allowed_levels, (list, tuple, set)) or not set(allowed_levels) <= {1,2,3}:
                raise ValueError('legal_source_levels must contain only 1, 2 or 3')
        eligible = np.array([i for i,d in enumerate(self.documents)
                             if int(d.get('priority_level', 3)) in (allowed_levels if allowed_levels is not None else (1,2,3))], dtype=int)
        if not len(eligible): return []
        query_ids = [self.vocabulary[t] for t in tokens(query_terms(query)) if t in self.vocabulary]
        scores = np.asarray(self.matrix[:,query_ids].sum(axis=1)).ravel() if query_ids else np.zeros(len(self.documents))
        maximum = float(scores[eligible].max())
        if maximum <= 0:
            # Unknown terms have no lexical support; preserve a bounded semantic fallback.
            pool = eligible
            relevance = np.asarray(similarities)
            reason = 'semantic_fallback_no_lexical_match'
        else:
            relevance = scores / maximum
            pool = eligible[np.argsort(-scores[eligible], kind='stable')][:max(40,top_k)]
            reason = 'lexical_evidence'
        priority = np.array([int(d.get('priority_level',3)) for d in self.documents])
        mixed = .7*relevance+.3*(1-gap*priority)
        selected = pool[np.argsort(-mixed[pool], kind='stable')][:top_k]
        return [{'idx':int(i),'similarity':float(similarities[i]),'relevance':float(relevance[i]),
                 'priority_level':int(priority[i]),'score':float(mixed[i]),'source':'legal',
                 'selection_reason':reason,'doc':self.documents[i]} for i in selected]
