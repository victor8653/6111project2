
import argparse
import nltk
from nltk.corpus import stopwords
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import spacy
import requests
from bs4 import BeautifulSoup

nltk.download('stopwords')
nltk.download('punkt')

#spaCy model
nlp = spacy.load("en_core_web_sm")

# Define a set of navigation/boilerplate words to filter out non-content entities
NAV_WORDS = {"home", "contact", "navigation", "upload", "community", "login", "signup", "donate", "about"}


def search_google(api_key, engine_id, query):
    """
    利用 Google Custom Search API 执行搜索，并返回结果。
    """
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=query, cx=engine_id).execute()
    return res


def download_webpage(url, timeout=10):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=timeout)
        if response.status_code == 200:
            return response.text
        else:
            print(f"Request to {url} returned status code: {response.status_code}")
    except Exception as e:
        print(f"Error while requesting {url}: {e}")
    return None

def extract_text(html):
    """
    Extract main content text from HTML. 
    """
    soup = BeautifulSoup(html, 'html.parser')
    # Remove script and style tags
    for tag in soup(["script", "style"]):
        tag.decompose()
    # Try to find <article> tag
    article = soup.find("article")
    if article:
        text = article.get_text(separator=" ", strip=True)
    else:
        # Try to find a div with id 'main'
        main_div = soup.find("div", id="main")
        if main_div:
            text = main_div.get_text(separator=" ", strip=True)
        else:
            # Fallback: extract all text
            text = soup.get_text(separator=" ", strip=True)
    return text[:10000]


#  spaCy footnote
def annotate_text(text):
    """
    Use spaCy to returns a list of tuples: (sentence, [(entity, label), ...])
    """
    doc = nlp(text)
    annotated_sentences = []
    # iterate each sentence
    for sent in doc.sents:
        entities = [(ent.text, ent.label_) for ent in sent.ents]
        annotated_sentences.append((sent.text, entities))
    return annotated_sentences


def is_valid_entity(entity):
    """
    Check if an entity is valid (not too short and not a common navigation word).
    """
    entity_lower = entity.lower().strip()
    if len(entity_lower) < 3:
        return False
    if entity_lower in NAV_WORDS:
        return False
    return True


def candidate_entity_pairs(annotations, relation):
    """
    Construct candidate entity pairs based on the target relation.
    
    Target relations:
      1,2 (Schools_Attended or Work_For): subject: PERSON, object: ORG (or GPE)
      3 (Live_In): subject: PERSON, object: GPE or LOC
      4 (Top_Member_Employees): subject: ORG, object: PERSON
      
    Returns a list of tuples: (sentence, subject, object)
    """
    candidate_pairs = []
    for sentence, entities in annotations:
        # Filter out invalid entities
        valid_entities = [(ent, label) for ent, label in entities if is_valid_entity(ent)]
        for i in range(len(valid_entities)):
            for j in range(i + 1, len(valid_entities)):
                e1, label1 = valid_entities[i]
                e2, label2 = valid_entities[j]
                if relation in [1, 2]:  # Schools_Attended or Work_For
                    if label1 == "PERSON" and label2 in ["ORG", "GPE"]:
                        candidate_pairs.append((sentence, e1, e2))
                    elif label1 in ["ORG", "GPE"] and label2 == "PERSON":
                        candidate_pairs.append((sentence, e2, e1))
                elif relation == 3:  # Live_In
                    if label1 == "PERSON" and label2 in ["GPE", "LOC"]:
                        candidate_pairs.append((sentence, e1, e2))
                    elif label2 == "PERSON" and label1 in ["GPE", "LOC"]:
                        candidate_pairs.append((sentence, e2, e1))
                elif relation == 4:  # Top_Member_Employees
                    if label1 == "ORG" and label2 == "PERSON":
                        candidate_pairs.append((sentence, e1, e2))
                    elif label1 == "PERSON" and label2 == "ORG":
                        candidate_pairs.append((sentence, e2, e1))
    return candidate_pairs


def extract_relation(candidate, method, confidence_threshold):
    """
    Call the relation extraction model on a candidate entity pair.
    This is a placeholder implementation; replace it with an actual call to SpanBERT or Gemini API.
    Returns a tuple: (subject, relation, object, confidence).
    If the extraction confidence is below the threshold (for SpanBERT), returns None.
    """
    sentence, subject, obj = candidate
    if method == "spanbert":
        # TODO: Replace with actual SpanBERT call
        extracted_relation = "per:employee_of"
        confidence = 0.95  # Example confidence
    elif method == "-gemini":
        # TODO: Replace with actual Gemini API call
        extracted_relation = "per:employee_of"
        confidence = 1.0
    else:
        return None

    if method == "spanbert" and confidence < confidence_threshold:
        return None
    return (subject, extracted_relation, obj, confidence)


def deduplicate_relations(relations):
    """
    Deduplicate the extracted relation tuples.
    For identical (subject, relation, object) tuples, only keep the one with the highest confidence.
    Returns a list of deduplicated tuples sorted by confidence in descending order.
    """
    dedup = {}
    for tup in relations:
        subject, rel, obj, conf = tup
        key = (subject.strip(), rel.strip(), obj.strip())
        if key not in dedup or dedup[key] < conf:
            dedup[key] = conf
    processed = [(key[0], key[1], key[2], dedup[key]) for key in dedup]
    processed.sort(key=lambda x: x[3], reverse=True)
    return processed




def main():
    parser = argparse.ArgumentParser(description="Project 2 - Iterative Information Extraction System")
    parser.add_argument("method", choices=["spanbert", "-gemini"], help="Extraction method: spanbert or -gemini")
    parser.add_argument("google_api_key", help="Google Custom Search API key")
    parser.add_argument("google_engine_id", help="Google Custom Search Engine ID")
    parser.add_argument("google_gemini_api_key", help="Google Gemini API key")
    parser.add_argument("r", type=int, choices=[1, 2, 3, 4],
                        help="Relation to extract (1: Schools_Attended, 2: Work_For, 3: Live_In, 4: Top_Member_Employees)")
    parser.add_argument("t", type=float, help="Extraction confidence threshold (used for spanbert)")
    parser.add_argument("q", help="Seed query, e.g., 'bill gates microsoft'")
    parser.add_argument("k", type=int, help="Number of tuples to extract")
    args = parser.parse_args()

    print("Parameters:")
    print(f"  Method           : {args.method}")
    print(f"  Google API Key   : {args.google_api_key}")
    print(f"  Engine ID        : {args.google_engine_id}")
    print(f"  Gemini API Key   : {args.google_gemini_api_key}")
    print(f"  Relation (r)     : {args.r}")
    print(f"  Threshold (t)    : {args.t}")
    print(f"  Seed Query (q)   : {args.q}")
    print(f"  Number of Tuples : {args.k}\n")

    max_iterations = 10
    iteration_count = 0
    # Store all extracted relation tuples
    extracted_relations = []
    # Track which tuples have been used as seeds for query expansion
    used_seeds = set()

    # Initial query from command line argument
    current_query = args.q.lower()

    while iteration_count < max_iterations:
        print(f"\n===== Iteration {iteration_count+1} =====")
        print(f"Current Query: {current_query}\n")

        try:
            res = search_google(args.google_api_key, args.google_engine_id, current_query)
        except Exception as e:
            print(f"Error during Google search: {e}")
            break

        new_extracted = []
        if "items" not in res:
            print("No search results returned.")
        else:
            for item in res.get("items", []):
                url = item.get("link")
                print(f"\nProcessing URL: {url}")
                html = download_webpage(url)
                if html:
                    text = extract_text(html)
                    print(f"Extracted text length: {len(text)}")
                    annotations = annotate_text(text)
                    candidates = candidate_entity_pairs(annotations, args.r)
                    print(f"Found {len(candidates)} candidate entity pairs.")
                    for candidate in candidates:
                        result = extract_relation(candidate, args.method, args.t)
                        if result is not None:
                            new_extracted.append(result)
                else:
                    print("Failed to download the webpage.")

        # Combine new extracted tuples with previous ones and deduplicate
        extracted_relations = deduplicate_relations(extracted_relations + new_extracted)
        print(f"\nTotal extracted relations so far: {len(extracted_relations)}")

        # Check if we have reached the desired number of tuples
        if len(extracted_relations) >= args.k:
            print("Desired number of tuples extracted. Stopping iterations.")
            break

        # Select an unused seed tuple from extracted_relations to generate a new query
        seed_tuple = None
        for tup in extracted_relations:
            key = (tup[0].strip(), tup[1].strip(), tup[2].strip())
            if key not in used_seeds:
                seed_tuple = tup
                used_seeds.add(key)
                break

        if seed_tuple is None:
            print("No new seed tuple found. ISE has stalled.")
            break

        # Construct a new query using the seed tuple's subject and object
        new_query = f"{seed_tuple[0]} {seed_tuple[2]}"
        current_query = new_query.lower()

        iteration_count += 1

    # Final output: deduplicate and take the top k tuples sorted by confidence
    final_relations = extracted_relations[:args.k]
    print("\n===== Final Extracted Relations =====")
    for tup in final_relations:
        subject, rel, obj, conf = tup
        print(f"[Subject: {subject}, Relation: {rel}, Object: {obj}, Confidence: {conf}]")
    if len(final_relations) < args.k:
        print(f"Warning: Only {len(final_relations)} tuples extracted, which is less than the desired {args.k}.")

if __name__ == "__main__":
    main()
