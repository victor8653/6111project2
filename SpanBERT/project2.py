import argparse
import time
from googleapiclient.discovery import build
import spacy
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai


from spanbert import SpanBERT
from spacy_help_functions import create_entity_pairs




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

def extract_text_spanbert(html):
    soup = BeautifulSoup(html, 'html.parser')

    paragraphs = soup.find_all('p')
    text = " ".join(p.get_text(separator=" ", strip=True) for p in paragraphs)
    text = text.replace('\n', ' ').replace('\t', '').replace('\xa0', '')
    return text[:10000]

def extract_text_gemini(html):
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


def is_valid_entity(entity):
    entity_lower = entity.lower().strip()
    return len(entity_lower) >= 3 and entity_lower not in NAV_WORDS



def filter_entity_pairs(entity_pairs, r):
    filtered_candidates = []

    for context, e1_info, e2_info in entity_pairs:
        e1_text, e1_label, e1_span = e1_info
        e2_text, e2_label, e2_span = e2_info

        # Relation 1: Schools_Attended
        if r == 1:
            if e1_label == "PERSON" and e2_label == "ORGANIZATION":
                filtered_candidates.append((context, e1_info, e2_info))
            elif e2_label == "PERSON" and e1_label == "ORGANIZATION":
                filtered_candidates.append((context, e2_info, e1_info))

        # Relation 2: Work_For
        elif r == 2:
            if e1_label == "PERSON" and e2_label == "ORGANIZATION":
                filtered_candidates.append((context, e1_info, e2_info))
            elif e2_label == "PERSON" and e1_label == "ORGANIZATION":
                filtered_candidates.append((context, e2_info, e1_info))

        # Relation 3: Live_In
        elif r == 3:
            if e1_label == "PERSON" and e2_label in ["LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]:
                filtered_candidates.append((context, e1_info, e2_info))
            elif e2_label == "PERSON" and e1_label in ["LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]:
                filtered_candidates.append((context, e2_info, e1_info))

        # Relation 4: Top_Member_Employees
        elif r == 4:
            if e1_label == "ORGANIZATION" and e2_label == "PERSON":
                filtered_candidates.append((context, e1_info, e2_info))
            elif e2_label == "ORGANIZATION" and e1_label == "PERSON":
                filtered_candidates.append((context, e2_info, e1_info))

    return filtered_candidates



def run_spanbert(example, r, spanbert_instance):
    preds = spanbert_instance.predict([example])
    if preds and len(preds) > 0:
        relation, confidence = preds[0]
        if (r == 1 and relation != 'per:schools_attended') or (r == 2 and relation != 'per:employee_of') or (
                r == 3 and (relation not in (
        'per:countries_of_residence', 'per:cities_of_residence', 'per:stateorprovinces_of_residence')) or (
                        r == 4 and relation != 'org:top_members/employees')):
            return "", 0.0
        return relation, confidence



def run_gemini(sentence, subject, obj, gemini_api_key, r):
    time.sleep(5)
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




def extract_relation(candidate, method, confidence_threshold, gemini_api_key, r, spanbert_instance=None):
    tokens, entity1, entity2 = candidate

    subject, subject_label, subject_span = entity1
    obj, obj_label, obj_span = entity2

    sentence = ' '.join(tokens)

    spanbert_ex ={"tokens": tokens, "subj": entity1, "obj": entity2}
    if method == "spanbert":
        relation, confidence = run_spanbert(spanbert_ex, r, spanbert_instance)
        if confidence < confidence_threshold:
            return None
    elif method == "gemini":
        relation, confidence = run_gemini(sentence, subject, obj, gemini_api_key, r)
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

def print_header(args, method):
    print("____")
    print("Parameters:")
    print("\tClient key\t= {}".format(args.google_api_key))
    print("\tEngine key\t= {}".format(args.google_engine_id))
    print("\tGemini key\t= {}".format(args.google_gemini_api_key))
    method_str = method
    print("\tMethod\t= {}".format(method_str))

    relation_names = {1: "Schools_Attended", 2: "Work_For", 3: "Live_In", 4: "Top_Member_Employees"}
    print("\tRelation\t= {}".format(relation_names.get(args.r, "Unknown")))
    print("\tThreshold\t= {}".format(args.t))
    print("\tQuery\t\t= {}".format(args.q.lower()))
    print("\t# of Tuples\t= {}".format(args.k))
    print("Loading necessary libraries; This should take a minute or so ...)")
    
def process_url(url, url_index, total_urls, args, spanbert_instance, method, res):
    print("\nURL ( {} / {}): {}".format(url_index, total_urls, url))

    item = res["items"][url_index - 1]

    skip_url = False
    if "fileFormat" in item and item["fileFormat"].lower() != "html":
        print(f"Skipping non-HTML file: {item.get('title')} with format {item['fileFormat']}")
        skip_url = True

    if skip_url:
        return None, 0


    print("\tFetching text from url ...")
    html = download_webpage(url)
    if not html:
        print("\tUnable to fetch URL. Continuing.")
        return None, 0


    if len(html) > 10000:
        print("\tTrimming webpage content from {} to 10000 characters".format(len(html)))
    print("\tWebpage length (num characters): {}".format(min(len(html), 10000)))
    print("\tAnnotating the webpage using spacy...")
    if method == "spanbert":
        text = extract_text_spanbert(html)
    else:
        text = extract_text_gemini(html)
    doc = nlp(text)
    num_sents = len([sent for sent in doc.sents])
    print("\tExtracted {} sentences. Processing each sentence one by one to check for presence of right pair of named entity types; if so, will run the second pipeline ...".format(num_sents))
    # 
    step = max(1, num_sents//5)
    for i in range(num_sents):
        if i % step == 0:
            print("\tProcessed {} / {} sentences".format(i, num_sents))
    unfiltered_candidates = []
    entities_of_interest = ["PERSON", "LOCATION", "CITY", "ORGANIZATION", "STATE_OR_PROVINCE", "COUNTRY"]
    for sents_doc in doc.sents:
        entity_pairs = create_entity_pairs(sents_doc, entities_of_interest)
        unfiltered_candidates.extend(entity_pairs)

    candidates = filter_entity_pairs(unfiltered_candidates, args.r)
    extracted = []
    for c in candidates:
        result = extract_relation(c, method, args.t, args.google_gemini_api_key, args.r, spanbert_instance)
        if result:
            extracted.append(result)
    print("\tRelations extracted from this website: {} (Overall: {})".format(len(extracted), len(extracted)))
    return extracted, num_sents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-spanbert", action="store_true", help="Use SpanBERT for relation extraction")
    parser.add_argument("-gemini", action="store_true", help="Use Gemini for relation extraction")
    parser.add_argument("google_api_key")
    parser.add_argument("google_engine_id")
    parser.add_argument("google_gemini_api_key")
    parser.add_argument("r", type=int, choices=[1, 2, 3, 4])
    parser.add_argument("t", type=float)
    parser.add_argument("q")
    parser.add_argument("k", type=int)

    args = parser.parse_args()

    if not (args.spanbert or args.gemini):
        print("Error: You must specify either '-spanbert' or '-gemini'.")
        return

    if args.spanbert:
        method = "spanbert"
    elif args.gemini:
        method = "gemini"

    
    print_header(args, method)

    # Initialize X, the set of extracted tuples
    extracted_relations = []

    spanbert_instance = None
    if method == "spanbert":
        print("Loading SpanBERT model...")
        spanbert_instance = SpanBERT("./pretrained_spanbert")

    print("=========== Iteration: 0 - Query: {} ===========".format(args.q.lower()))
    query = args.q.lower()
    # extracted_relations = []
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
            rels, _ = process_url(url, idx, total_urls, args, spanbert_instance, method, res)
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
        len(extracted_relations)))
    for s, r, o, c in extracted_relations:
        if method == "spanbert":
            print("Confidence: {:.8f} \t\t| Subject: {} \t\t| Object: {}".format(c, s, o))
        else: print("Subject: {} \t\t| Object: {}".format( s, o))

    print("Total # of iterations = {}".format(total_iterations))

if __name__ == "__main__":
    main()

