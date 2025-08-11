import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')
from typing import List, Dict, Optional, TypedDict, Tuple, Any
import openai
import json
import os
from nltk.tokenize import sent_tokenize
from dotenv import load_dotenv

load_dotenv()

# Load local embedding model lazily
_model = None

def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    return _model

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english')) - {'using', 'with'}

def _get_wordnet_pos(tag: str) -> str:
    tag_dict = {
        'J': wordnet.ADJ,
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'R': wordnet.ADV
    }
    return tag_dict.get(tag[0].upper(), wordnet.NOUN)

# Keyword lists (could be externalized later)

class SkillMatchDict(TypedDict):
    category: str
    item: str
    sentence: str
    similarity: float
    justification: str

class SectionScores(TypedDict):
    hard_skills_score: float
    responsibilities_score: float
    qualifications_score: float
    soft_skills_score: float
    global_score: float

class ExtraMatchInfo(TypedDict, total=False):
    years_experience_found: Dict[str, int]
    location_found: Optional[str]
    missing_soft_skills: List[str]

def extract_sentences(text: str) -> List[str]:
    return sent_tokenize(text)

def extract_location(text: str) -> str:
    pattern = r"(remote|onsite|hybrid|[A-Z][a-z]+,? [A-Z][a-z]+)"
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1) if match else ""



def extract_skills_from_text(text_blocks: List[str]) -> Tuple[List[str], List[str]]:
    

    combined_text = ' '.join(text_blocks)

    prompt = (
    "Extract all skills mentioned in the following job description. "
    "For each skill, classify it as either a HARD SKILL or a SOFT SKILL. "
    "Return the result as a JSON list of objects like: {\"skill\": \"...\", \"type\": \"hard\" or \"soft\"}.\n\n"
    f"Job description:\n{combined_text}"
    )

    try:
        api_key = os.getenv("OPENAI_API_KEY")
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=800
        )

        print("Content returned from ChatGPT.", response)
        response_content = response.choices[0].message.content.strip()
        json_text = re.search(r"\[.*\]", response_content, re.DOTALL)

        if json_text:
            parsed = json.loads(json_text.group())
        else:
            raise ValueError("No JSON array found in GPT response.")    
    
        hard_skills = [entry["skill"] for entry in parsed if entry["type"].lower() == "hard"]
        print("Hard skills extracted:", hard_skills)
        soft_skills = [entry["skill"] for entry in parsed if entry["type"].lower() == "soft"]
        return list(set(hard_skills)), list(set(soft_skills))
    except Exception as e:
            print("Failed to extract skills with ChatGPT:", e)
            return [], []


def extract_soft_skills(text: str, soft_skills: List[str]) -> List[str]:
    text = re.sub(r'\s+', ' ', text.lower())
    return [s for s in soft_skills if s.lower() in text]


def evaluate_resume_against_job(
    resume_text: str,
    job_title: str,
    responsibilities: List[str],
    qualifications_and_experience: List[str],
    required_location: Optional[str] = None
) -> Tuple[List[SkillMatchDict], SectionScores, ExtraMatchInfo]:

    job_blocks = responsibilities + qualifications_and_experience
    hard_skills, soft_skills = extract_skills_from_text(job_blocks)

    resume_sentences = extract_sentences(resume_text)
    matches: List[SkillMatchDict] = []
    scores = {
        "hard_skills_score": 0.0,
        "responsibilities_score": 0.0,
        "qualifications_score": 0.0,
        "soft_skills_score": 0.0,
        "global_score": 0.0
    }
    extra: ExtraMatchInfo = {
        "years_experience_found": {},
        "location_found": extract_location(resume_text),
        "missing_soft_skills": []
    }

    def score_section(category: str, items: List[str]) -> float:
        hits = 0
        for item in items:
            result = find_best_sentence_match(resume_sentences, item)
            if result:
                sentence, sim = result
                matches.append({
                    "category": category,
                    "item": item,
                    "sentence": sentence,
                    "similarity": float(sim),
                    "justification": f"Matched with: '{sentence}'"
                })
                hits += 1
        return round(hits / len(items), 2) if items else 0.0

    scores["hard_skills_score"] = score_section("hard_skill", hard_skills)
    scores["responsibilities_score"] = score_section("responsibility", responsibilities)
    scores["qualifications_score"] = score_section("qualification", qualifications_and_experience)
    scores["soft_skills_score"] = score_section("soft_skill", soft_skills)

    for skill in hard_skills:
        extra["years_experience_found"][skill] = extract_years_experience(resume_text, skill)

    scores["global_score"] = round(
        scores["hard_skills_score"] * 0.4 +
        scores["responsibilities_score"] * 0.25 +
        scores["qualifications_score"] * 0.25 +
        scores["soft_skills_score"] * 0.1,
        2
    )

    return matches, scores, extra


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    words = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    lemmatized = [lemmatizer.lemmatize(w, _get_wordnet_pos(pos)) for w, pos in pos_tags]
    return ' '.join([w for w in lemmatized if w not in stop_words])


def find_best_sentence_match(sentences, target, threshold=0.6):
    from sklearn.metrics.pairwise import cosine_similarity

    target_norm = normalize_text(target)
    target_embedding = get_model().encode(target_norm)
    best_score = 0.0
    best_sentence = None

    for sentence in sentences:
        sentence_norm = normalize_text(sentence)
        sentence_embedding = get_model().encode(sentence_norm)
        sim = cosine_similarity([target_embedding], [sentence_embedding])[0][0]
        if sim > best_score:
            best_score = sim
            best_sentence = sentence

    if best_score > threshold or target_norm in normalize_text(best_sentence or ""):
        return best_sentence, best_score
    return None


def extract_years_experience(text: str, keyword: str) -> int:
    text = text.lower()
    keyword = keyword.lower()
    pattern = rf"(\d+)\+?\s+(years|años)\s+of\s+experience.*?{keyword}"
    matches = re.findall(pattern, text)
    if matches:
        return int(matches[0][0])
    return 0
