import argparse
import nltk
from nltk.corpus import stopwords
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import spacy
import requests
from bs4 import BeautifulSoup
import json

# 尝试导入 newspaper3k，如果没有安装，可注释掉相关逻辑
try:
    from newspaper import Article
except ImportError:
    Article = None

# 尝试导入 Google Gemini API
try:
    import google.generativeai as palm
except ImportError:
    print("Please install google-generativeai: pip install -q -U google-generativeai")
    palm = None

# 从 SpanBERT 仓库导入 SpanBERT 类
from SpanBERT.spanbert import SpanBERT



nltk.download('stopwords')
nltk.download('punkt')

nlp = spacy.load("en_core_web_lg")

NAV_WORDS = {"home", "contact", "navigation", "upload", "community", "login", "signup", "donate", "about"}

def search_google(api_key, engine_id, query):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=query, cx=engine_id).execute()
    return res

def download_webpage(url, timeout=10):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=timeout)
        if response.status_code == 200:
            return response.text
    except Exception as e:
        print(f"Error while requesting {url}: {e}")
    return None

def extract_text(html):
    """
    改进后的文本提取：尝试 newspaper3k、readability-lxml、然后回退到 BeautifulSoup。
    """
    text = ""
    # 尝试 newspaper3k 提取
    try:
        if Article is not None:
            article = Article(url="")
            article.set_html(html)
            article.parse()
            text = article.text
    except Exception:
        pass

    # 如果 newspaper3k 结果不足，则尝试 readability-lxml
    if not text or len(text) < 200:
        try:
            from readability import Document
            doc = Document(html)
            extracted_html = doc.summary()
            text = BeautifulSoup(extracted_html, 'html.parser').get_text(separator=" ", strip=True)
        except Exception:
            pass

    # 如果仍然不足，则使用 BeautifulSoup
    if not text or len(text) < 200:
        soup = BeautifulSoup(html, 'html.parser')
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            tag.decompose()
        article_tag = soup.find("article")
        if article_tag:
            text = article_tag.get_text(separator=" ", strip=True)
        else:
            main_div = soup.find("div", id="main") or soup.find("div", class_="content")
            if main_div:
                text = main_div.get_text(separator=" ", strip=True)
            else:
                text = soup.get_text(separator=" ", strip=True)

    text = " ".join(text.split())
    return text[:10000]

def annotate_text(text):
    doc = nlp(text)
    annotated_sentences = []
    for sent in doc.sents:
        s = sent.text.strip()
        if len(s) < 20:
            continue
        lower_s = s.lower()
        if sum(1 for nav in NAV_WORDS if nav in lower_s) > 1:
            continue
        entities = [(ent.text, ent.label_) for ent in sent.ents]
        annotated_sentences.append((s, entities))
    return annotated_sentences

def is_valid_entity(entity):
    entity_lower = entity.lower().strip()
    return len(entity_lower) >= 3 and entity_lower not in NAV_WORDS

def candidate_entity_pairs(annotations, relation):
    pairs = []
    for sentence, entities in annotations:
        valid = [(ent, label) for ent, label in entities if is_valid_entity(ent)]
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                e1, l1 = valid[i]
                e2, l2 = valid[j]
                if relation in [1, 2]:  # Schools_Attended / Work_For
                    if l1 == "PERSON" and l2 in ["ORG", "GPE"]:
                        pairs.append((sentence, e1, e2))
                    elif l2 == "PERSON" and l1 in ["ORG", "GPE"]:
                        pairs.append((sentence, e2, e1))
                elif relation == 3:  # Live_In
                    if l1 == "PERSON" and l2 in ["GPE", "LOC"]:
                        pairs.append((sentence, e1, e2))
                    elif l2 == "PERSON" and l1 in ["GPE", "LOC"]:
                        pairs.append((sentence, e2, e1))
                elif relation == 4:  # Top_Member_Employees
                    if l1 == "ORG" and l2 == "PERSON":
                        pairs.append((sentence, e1, e2))
                    elif l2 == "ORG" and l1 == "PERSON":
                        pairs.append((sentence, e2, e1))
    return pairs

def run_spanbert(sentence, subject, obj, spanbert_instance):
    """
    简化示例：仅将 subject 视为 tokens[0]，object 视为 tokens[-1]。
    建议你使用 spaCy 的 tokenization 来定位 subject、object 在句子中的精确位置。
    """
    tokens = sentence.split()
    subj_idx = 0
    obj_idx = len(tokens) - 1
    # 这里假设 subject 是 PERSON, object 是 ORG，你需要根据 relation 类型来动态决定
    example = {
        'tokens': tokens,
        'subj': (subject, "PERSON", (subj_idx, subj_idx + 1)),
        'obj': (obj, "ORGANIZATION", (obj_idx, obj_idx + 1))
    }
    preds = spanbert_instance.predict([example])
    if preds and len(preds) > 0:
        relation, confidence = preds[0]
        return relation, confidence
    return "", 0.0

def run_gemini(sentence, subject, obj, gemini_api_key):
    if palm is None:
        return "", 0.0
    palm.configure(api_key=gemini_api_key)
    prompt = (
        f"Extract the relation between the following entities in the sentence. "
        f"Sentence: \"{sentence}\". "
        f"Subject: \"{subject}\". "
        f"Object: \"{obj}\". "
        f"Return a JSON with keys 'relation' and 'confidence'."
    )
    try:
        response = palm.generate_text(prompt=prompt, model="models/gemini-2", temperature=0.0)
        result = json.loads(response.result)
        relation = result.get("relation", "")
        confidence = float(result.get("confidence", 0.0))
        return relation, confidence
    except Exception as e:
        print(f"Error in Gemini API call: {e}")
        return "", 0.0

def extract_relation(candidate, method, confidence_threshold, gemini_api_key, spanbert_instance=None):
    sentence, subject, obj = candidate
    if method == "spanbert":
        relation, confidence = run_spanbert(sentence, subject, obj, spanbert_instance)
        if confidence < confidence_threshold:
            return None
    elif method == "-gemini":
        relation, confidence = run_gemini(sentence, subject, obj, gemini_api_key)
    else:
        return None
    return (subject, relation, obj, confidence)

def deduplicate_relations(relations):
    seen = {}
    for s, r, o, c in relations:
        key = (s.strip(), r.strip(), o.strip())
        if key not in seen or seen[key] < c:
            seen[key] = c
    # 按置信度降序
    return sorted([(k[0], k[1], k[2], seen[k]) for k in seen], key=lambda x: x[3], reverse=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=["spanbert", "-gemini"])
    parser.add_argument("google_api_key")
    parser.add_argument("google_engine_id")
    parser.add_argument("google_gemini_api_key")
    parser.add_argument("r", type=int, choices=[1, 2, 3, 4])
    parser.add_argument("t", type=float)
    parser.add_argument("q")
    parser.add_argument("k", type=int)
    args = parser.parse_args()

    # 如果 method 是 spanbert，则加载预训练的 SpanBERT 模型
    spanbert_instance = None


    if args.method == "spanbert":
        print("Loading SpanBERT model...")
        spanbert_instance = SpanBERT("SpanBERT/pretrained_spanbert")
        from transformers import BertTokenizer
        spanbert_instance.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")


    print("Starting Iterative Extraction...")
    query = args.q.lower()
    extracted = []
    used = set()

    for iteration in range(10):
        print(f"\nIteration {iteration+1}: Query = {query}")
        try:
            res = search_google(args.google_api_key, args.google_engine_id, query)
        except Exception as e:
            print(f"Search error: {e}")
            break

        new_relations = []
        if "items" not in res:
            print("No search results.")
            continue
        for item in res["items"]:
            url = item.get("link")
            print(f"Processing: {url}")
            html = download_webpage(url)
            if not html:
                continue
            text = extract_text(html)
            annots = annotate_text(text)
            candidates = candidate_entity_pairs(annots, args.r)
            for c in candidates:
                result = extract_relation(c, args.method, args.t, args.google_gemini_api_key, spanbert_instance)
                if result:
                    new_relations.append(result)

        extracted = deduplicate_relations(extracted + new_relations)
        print(f"Total extracted so far: {len(extracted)}")
        if len(extracted) >= args.k:
            break

        # 选择一个新的种子元组来生成下一次查询
        for s, r, o, _ in extracted:
            key = (s.strip(), r.strip(), o.strip())
            if key not in used:
                query = f"{s} {o}".lower()
                used.add(key)
                break
        else:
            print("No new seed found. Stopping.")
            break

    print("\nFinal Extracted Relations:")
    for s, r, o, c in extracted[:args.k]:
        print(f"[Subject: {s}, Relation: {r}, Object: {o}, Confidence: {c:.2f}]")

if __name__ == "__main__":
    main()
