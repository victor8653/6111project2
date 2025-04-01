# temporary last version 31 6:30pm
import argparse
import nltk
from nltk.corpus import stopwords
from googleapiclient.discovery import build
# from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import spacy
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from SpanBERT.spanbert import SpanBERT

# Try importing newspaper3k. If not installed, related logic will be skipped.
try:
    from newspaper import Article
except ImportError:
    Article = None



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
    # Try extracting with newspaper3k
    try:
        if Article is not None:
            article = Article(url="")
            article.set_html(html)
            article.parse()
            text = article.text
    except Exception:
        pass

    # If newspaper3k result is insufficient, try readability-lxml
    if not text or len(text) < 200:
        try:
            from readability import Document
            doc = Document(html)
            extracted_html = doc.summary()
            text = BeautifulSoup(extracted_html, 'html.parser').get_text(separator=" ", strip=True)
        except Exception:
            pass

    # If still insufficient, use BeautifulSoup
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
                if relation == 2:  # Work_For
                    # only PERSON and ORG 
                    if l1 == "PERSON" and l2 == "ORG":
                        pairs.append((sentence, e1, e2))
                    elif l2 == "PERSON" and l1 == "ORG":
                        pairs.append((sentence, e2, e1))
                elif relation == 1:  # Schools_Attended and GPE or ORG
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
    # use spaCy for sentences
    doc = nlp(sentence)
    
    subj_start = sentence.find(subject)
    obj_start = sentence.find(obj)
    
    if subj_start == -1 or obj_start == -1:
        return "", 0.0
    
    subj_end = subj_start + len(subject)
    obj_end = obj_start + len(obj)
    
    subj_span = doc.char_span(subj_start, subj_end)
    obj_span = doc.char_span(obj_start, obj_end)
    if subj_span is None or obj_span is None:
        return "", 0.0
    
    subj_token_start = subj_span.start
    subj_token_end = subj_span.end
    obj_token_start = obj_span.start
    obj_token_end = obj_span.end
    
    tokens = [token.text for token in doc]
    
    example = {
        'tokens': tokens,
        'subj': (subject, "PERSON", (subj_token_start, subj_token_end)),
        'obj': (obj, "ORGANIZATION", (obj_token_start, obj_token_end))
    }
    
    preds = spanbert_instance.predict([example])
    if preds and len(preds) > 0:
        relation, confidence = preds[0]
        return relation, confidence

    print("Tokens:", tokens)
    print("Subject span:", subj_span.start, subj_span.end)
    print("Object span:", obj_span.start, obj_span.end)

    return "", 0.0


def run_gemini(sentence, subject, obj, gemini_api_key):
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    generation_config = genai.types.GenerationConfig(
    max_output_tokens=200,
    temperature=0.2,
    top_p=1,
    top_k=32
    )
    prompt = f"""
        Given the sentence:
        \"{sentence}\"

        What is the relation between \"{subject}\" and \"{obj}\"? 
        Only answer with the relation name if it exists, otherwise say "no_relation".
        """

    try:
        response = model.generate_content(prompt, generation_config=generation_config)
        result = response.text.strip().lower()
        return result, 1.0  
    except Exception as e:
        print("Error in Gemini API call:", e)
        return "no_relation", 0.0


def extract_relation(candidate, method, confidence_threshold, gemini_api_key, spanbert_instance=None):
    sentence, subject, obj = candidate
    if method == "spanbert":
        relation, confidence = run_spanbert(sentence, subject, obj, spanbert_instance)
        if confidence < confidence_threshold:
            return None
    elif method == "gemini":
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
    # 
    return sorted([(k[0], k[1], k[2], seen[k]) for k in seen], key=lambda x: x[3], reverse=True)

def print_header(args):
    print("Loading pre-trained spanBERT from ./pretrained_spanbert\n")
    print("____")
    print("Parameters:")
    print("\tClient key\t= {}".format(args.google_api_key))
    print("\tEngine key\t= {}".format(args.google_engine_id))
    print("\tGemini key\t= {}".format(args.google_gemini_api_key))
    method_str = args.method
    print("\tMethod\t= {}".format(method_str))

    relation_names = {1: "Schools_Attended", 2: "Work_For", 3: "Live_In", 4: "Top_Member_Employees"}
    print("\tRelation\t= {}".format(relation_names.get(args.r, "Unknown")))
    print("\tThreshold\t= {}".format(args.t))
    print("\tQuery\t\t= {}".format(args.q.lower()))
    print("\t# of Tuples\t= {}".format(args.k))
    print("Loading necessary libraries; This should take a minute or so ...)")
    
def process_url(url, url_index, total_urls, args, spanbert_instance):
    print("\nURL ( {} / {}): {}".format(url_index, total_urls, url))
    print("\tFetching text from url ...")
    html = download_webpage(url)
    if not html:
        print("\tUnable to fetch URL. Continuing.")
        return None, 0


    if len(html) > 10000:
        print("\tTrimming webpage content from {} to 10000 characters".format(len(html)))
    print("\tWebpage length (num characters): {}".format(min(len(html), 10000)))
    print("\tAnnotating the webpage using spacy...")
    text = extract_text(html)
    annots = annotate_text(text)
    print("\tExtracted {} sentences. Processing each sentence one by one to check for presence of right pair of named entity types; if so, will run the second pipeline ...".format(len(annots)))
    # 
    step = max(1, len(annots)//5)
    for i in range(len(annots)):
        if i % step == 0:
            print("\tProcessed {} / {} sentences".format(i, len(annots)))
    # 
    candidates = candidate_entity_pairs(annots, args.r)
    extracted = []
    for c in candidates:
        result = extract_relation(c, args.method, args.t, args.google_gemini_api_key, spanbert_instance)
        if result:
            extracted.append(result)
    print("\tRelations extracted from this website: {} (Overall: {})".format(len(extracted), len(extracted)))
    return extracted, len(annots)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=["spanbert", "gemini"])
    parser.add_argument("google_api_key")
    parser.add_argument("google_engine_id")
    parser.add_argument("google_gemini_api_key")
    parser.add_argument("r", type=int, choices=[1, 2, 3, 4])
    parser.add_argument("t", type=float)
    parser.add_argument("q")
    parser.add_argument("k", type=int)
    args = parser.parse_args()

    
    print_header(args)

    # Load SpanBERT model if using spanbert method
    spanbert_instance = None
    if args.method == "spanbert":
        print("Loading SpanBERT model...")
        spanbert_instance = SpanBERT("SpanBERT/pretrained_spanbert")
        from transformers import BertTokenizer
        spanbert_instance.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    print("=========== Iteration: 0 - Query: {} ===========".format(args.q.lower()))
    query = args.q.lower()
    extracted_relations = []
    used = set()

    # record used URL
    processed_urls = set()


    total_iterations = 0
    # 10 times
    for iteration in range(10):
        total_iterations = iteration + 1
        print("\n=========== Iteration: {} - Query: {} ===========".format(iteration, query))
        try:
            res = search_google(args.google_api_key, args.google_engine_id, query)
        except Exception as e:
            print("Search error: {}".format(e))
            break

        if "items" not in res:
            print("No search results.")
            break

        new_relations = []
        urls = [item.get("link") for item in res["items"]]
        total_urls = len(urls)

        for idx, url in enumerate(urls, 1):
            if url in processed_urls:
                continue
            
            processed_urls.add(url)
            rels, _ = process_url(url, idx, total_urls, args, spanbert_instance)
            if rels:
                new_relations.extend(rels)

        extracted_relations = deduplicate_relations(extracted_relations + new_relations)
        print("Total extracted so far: {}".format(len(extracted_relations)))
        if len(extracted_relations) >= args.k:
            break

        # new seed tuple for search
        for s, r, o, _ in extracted_relations:
            key = (s.strip(), r.strip(), o.strip())
            if key not in used:
                query = "{} {}".format(s, o).lower()
                used.add(key)
                break
        else:
            print("No new seed found. Stopping.")
            break

    print("\n================== ALL RELATIONS for {} ( {} ) =================".format(
        {1: "per:schools_attended", 2: "per:employee_of", 3: "per:cities_of_residence", 4: "org:top_members/employees"}.get(args.r, "unknown"),
        args.k))
    for s, r, o, c in extracted_relations[:args.k]:
        print("Confidence: {:.8f} \t\t| Subject: {} \t\t| Object: {}".format(c, s, o))
    print("Total # of iterations = {}".format(total_iterations))

if __name__ == "__main__":
    main()

