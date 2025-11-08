#!/usr/bin/env python3
"""
Offline cleaner for person-name columns in processed_with_people_emotion.parquet ONLY.
Reads processed_with_people_emotion.parquet, removes stopwords and non-name tokens from people strings,
and writes cleaned_sampled_data.parquet to final_data folder.

Usage:
  python clean_people_names.py \
    --sampled data_storage/processed_data/processed_with_people_emotion.parquet \
    --outdir data_storage/final_data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Set, Dict
import re

import pandas as pd

# Allow importing helpers from streamlit_app/
ROOT = Path(__file__).resolve().parents[2]  # .../capstone/capstone
STREAMLIT_APP_DIR = ROOT / "streamlit_app"
if str(STREAMLIT_APP_DIR) not in sys.path:
    sys.path.insert(0, str(STREAMLIT_APP_DIR))

try:
    # Reuse your existing robust name helpers
    from data_processors import extract_clean_names, is_likely_person_name  # type: ignore
except Exception:
    extract_clean_names = None
    is_likely_person_name = None

# Try to import the large EXCLUDE list from attribution.py to enrich exclusions
try:
    sys.path.insert(0, str(ROOT / "model_development"))
    from attribution import EXCLUDE_LOWER as ATTRIB_EXCLUDES  # type: ignore
except Exception:
    ATTRIB_EXCLUDES = set()


def _load_stopwords() -> Set[str]:
    """Load English stopwords plus domain extras. Falls back to a minimal set if NLTK not available."""
    extras = {
        # Common domain terms you'd like to strip from 'people' fields
        "network", "report", "study", "analysis", "article", "news",
        "group", "foundation", "society", "association", "center",
        "centre", "clinic", "hospital", "institute", "university",
        "college", "company", "organization", "government", "policy",
        "health", "care", "public", "private", "state", "federal",
        "research", "data", "information",
        # User-observed non-person tokens (seed list; case-insensitive)
        "advocacy", "alliance", "lobbying", "numerous", "officer",
        "starts", "advocate", "heath",
        # From scan of cleaned data
        "posted", "denver", "therapy", "med", "environ", "law", "hubs",
        "healthy", "gen", "disease", "initiative", "virus", "rotavirus",
        "recipes", "images", "max",
        # Requested specific exclusions
        "covid", "covid-19", "coronavirus", "sars-cov-2", "pandemic", "vaccine", "vaccines",
        "advil", "tylenol", "ibuprofen", "acetaminophen", "aspirin",
        "pfizer", "moderna", "johnson", "johnson&johnson", "johnson-and-johnson",
        # Animals and generic species terms
        "eagle", "eagles", "bear", "wolf", "wolves", "lion", "tiger", "dog", "cat", "species", "animal", "animals",
        # Additional user-provided explicit non-names (lowercased)
        "abc", "abc aviation", "academic scholarship", "acad diet", "acad sci",
        "ac menthol", "accuracy", "accura", "adweek", "adzippy", "afford",
        "afforddable", "affordable packages", "afluria", "diagnosis", "builds",
        "builsd", "alcohol", "alco", "alcatrz", "store", "influenza",
        "neuroscience", "altmetric", "alternatodiol",
        # Brands/products spotted
        "lactaid", "lactaid milk",
        "humanized mouse", "kitchen cabinet", "strawberry lemonade",
        "cherry lemonade", "cherry limeade", "cherry limeaid",
        "strawberry", "lemonade", "limeade", "limeaid",
        "abc", "abc aviation", "adweek", "advertise", "advertisement",
        "advertising", "accesswire", "accuweather", "ace therapeutics",
        "accuracy", "accura", "roofing",
        "sweat economy","sweat wallet","blockchain lab","app","milk","almond","maid zero","cherry",
        "outbreak","manchester united","meets civic","bug barometer","omaha packing","pmaha packing",
        "hamilton hall","deep cuts","chicken tortilla soup","enchilada soup","nacho soup","thai soup",
        "chicken tortilla","soup",
    }
    try:
        from nltk.corpus import stopwords  # type: ignore
        try:
            sw = set(stopwords.words("english"))
        except LookupError:
            # If stopwords not downloaded, fall back to a minimal set
            sw = {
                "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
                "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
                "been", "being", "this", "that", "these", "those", "it", "its",
            }
    except Exception:
        sw = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
            "been", "being", "this", "that", "these", "those", "it", "its",
        }
    return (sw | extras | set(map(str.lower, ATTRIB_EXCLUDES)))


US_STATES = {
    "alabama","alaska","arizona","arkansas","california","colorado","connecticut","delaware","florida","georgia",
    "hawaii","idaho","illinois","indiana","iowa","kansas","kentucky","louisiana","maine","maryland","massachusetts",
    "michigan","minnesota","mississippi","missouri","montana","nebraska","nevada","new hampshire","new jersey",
    "new mexico","new york","north carolina","north dakota","ohio","oklahoma","oregon","pennsylvania","rhode island",
    "south carolina","south dakota","tennessee","texas","utah","vermont","virginia","washington","west virginia",
    "wisconsin","wyoming","district of columbia","washington dc","d.c.","dc"
}

COMMON_COUNTRIES = {
    "united states","usa","us","canada","mexico","china","japan","south korea","north korea","india","russia",
    "united kingdom","uk","england","scotland","wales","ireland","france","germany","italy","spain","portugal",
    "brazil","argentina","chile","colombia","peru","australia","new zealand","south africa","nigeria","kenya",
    "egypt","israel","palestine","iran","iraq","syria","lebanon","turkey","saudi arabia","uae","qatar"
}

LOCATION_SUFFIXES = (
    " city"," county"," province"," state"," region"," parish"," borough"," district"," village",
    " island"," islands"," park"," river"," mountain"," mountains"," lake"," airport"
)

# Domain keyword sets (lowercased)
FOOD_KEYWORDS = {
    "soup","salad","burger","pizza","taco","enchilada","nacho","burrito","curry","noodle","noodles",
    "sandwich","coffee","tea","soda","juice","milk","almond","oat","yogurt","cheese","butter","bread",
    "lemonade","limeade","limeaid","strawberry","cherry","fried","chicken","beef","pork","seafood"
}

BEVERAGE_KEYWORDS = {
    "beer","wine","whiskey","vodka","gin","rum","tequila","cocktail","martini","latte","espresso","cappuccino"
}

BRAND_SUFFIXES = {
    "inc","corp","ltd","llc","gmbh","sa","plc","pty","co","co.","company","pharma","therapeutics",
    "biosciences","biotech","technologies","holdings","laboratories","laboratory","labs","lab","platform"
}

# Common drug/therapy suffixes (heuristic; lowercased)
DRUG_SUFFIXES = (
    "mab","nib","vir","vax","vaccine","pril","sartan","olol","cillin","cycline","cycline","azole",
    "avir","grel","oxetine","oxetine","xaban","floxacin","vastatin","afil","gliptin","gliflozin","tide"
)

# Substring-based non-person indicators (lowercased)
EXCLUSION_SUBSTRINGS = (
    "academy", "academic", "scholarship", "aviation", "packages", "store",
    "diagnosis", "influenza", "neuroscience", "metric", "altmetric",
    "menthol", "therapeutics", "pharma", "roofing", "weather", "screening",
    "advertis", "adweek", "adzippy", "mouse", "cabinet", "lemonade", "limeade",
    "limeaid",  # misspelling seen in data
    "kitchen", "hospital", "viewpoint", "wallet", "economy", "blockchain",
    "outbreak", "united", "barometer", "packing", "hall", "soup"
)

# Explicit multi-word brand/term exclusions (lowercased)
EXPLICIT_EXCLUDES = {
    # From user-provided lists (lowercased) - ALL words user specified
    "viewpoint", "humanized mouse", "hospital", "covid-19", "covid", "include", "Include", "yellow dye", "red dye", "sharks",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "monday's", "tuesday's", "wednesday's", "thursday's", "friday's", "saturday's", "sunday's",
    "measles", "measles vaccine", "measles cases", "measles surge", "measles outbreak", "measles vaccine", "measles cases", "measles surge", "measles outbreak",
    "diasorian spa", "hulu", "miami", "market research", "market research firm", "market research firms", "market research company", "market research companies",
     "market research report", "market research reports", "market research study", "market research studies", "market research analysis", "market research analyses", "market research report", "market research reports", "market research study", "market research studies", "market research analysis", "market research analyses",
    "google", "gemini", "netflix", "twitter", "instagram", "facebook", "x", "sprouted mat", "r- oceanside", 
    "parks", "rally planned", "asthma", "asthma cases", "asthma surge", "asthma outbreak", "asthma vaccine", "asthma cases", "asthma surge", "asthma outbreak",
    "adults", "gyms", "southeast promise", "zell", "venmo", "smokehouse barbeque", "creek", "cigar", "Advancing Pulmonary", "Money Box",
    "tanita", "invicta", "appendix supplementary", "dibley mj", "jin k", "mihrshahi", "ding trends", "int obes lond",
    "shipt shipt", "grow starter", "story", "vaccinia intravenous", "call", "veteran", "woodbridge", "meter", "joint statement",
    "tanita","invicta","appendix supplementary","dibley mj","jin k","mihrshahi","ding trends","int obes lond",
    "ahmed","ziviani","khan","d'este c drivers","delavari","sønderlund","soderlund",
    "swinburn effectiveness","renzaho acculturation","gibbs","magarey","hayes aj","kelly pj","frohlich kl",
    "potvin transcending","release","june","siekmann development","obes pract","patrick","alexander jj","rajmil",
    "mellor","boulton k","laird","lambert mg","eades","hartono","kinfu","sjöholm p","sjoholm p","davison",
    "juonala","singh socioeconomic","o'brien ks","obrien ks","liu","gibson k definitions","ideas grant",
    "crackdowns","clinical trial","clinical trials","respiratory disorder","respiratory disorders","respiatory disorder","respiatory disorders",
    "deep dive","deep dieve","lesbian","aids outreach","birmingham","achieving","creative biolabs","astrocyte",
    "ctexli","fluid delivery","mercedes ben's","mercedes benz","garlic supplements","garlic","allicin","song k",
    "cooked garlic heat","gen zeer","dawn","amaryllis","kick","cunt","mychart","bipap","kindness","mum",
    "geralthegrey","a+ grade","sure bert","slammed","robert lurie children's photograms","principles","rest",
    "obese","headache","muscle relaxants","runny nose","rash","survival","daughter","lymphocyte",
    "lipoprotien ratios","lipoprotein ratios","bald","ixchiq","catholic bishop","aci africa",
    "tigray people's liberation front","laughs","casa puebla","clemson","fliet","cyanotoxins","trader joe's",
    "famous yeti's","famous yeti","npr","americorps","yogi","head bolonga","produce rx","microbiol","acta",
    "francevill bp","dakar bp","hepatocystis spp","maximum likelihood","gamma","save gabson's tissue kid",
    "subtree pruning regrafting primatol","zool","decol","boundenga","worchester","ib","food stamps",
    "brick springs","radioligand therapy","ya numb","gmail","monocytogenes","aurobindo","vitamin k","vitamin",
    "vitamin c","echeverri","children can","approval mondavi","cale","healthy neighborhhoods","healthy people",
    "berry","besides berry","suite","gym hums","abiomed","accurary","atlantis strength",
    "emergency use authorizations","competitive landscape","lemon pepper wing","crispy","magic eraser",
    "generate aerosol",
    # Additional single words that are NOT names
    "posted", "denver", "therapy", "med", "environ", "law", "hubs", "healthy", "gen", "disease", 
    "initiative", "virus", "rotavirus", "recipes", "images", "max", "bill",  # standalone "bill" when not part of full name
    # Latest comprehensive exclusion list from user
    "volunteer", "nasal sprays", "of", "is", "took", "with", "not", "spiraled", "remains", "currently", 
    "providers", "enough", "frankenstein", "nitazene", "strain", "act", "boys", "campaigns", "dtap", "oval", 
    "wabc e", "patriot", "honey", "bologna", "bavarian nordic", "phone", "deep woods", "racine", "repel", 
    "amblyomma", "powassan", "madison", "product f", "product k", "nawrocki cc", "hinckley af", 
    "ticks borne dis", "emerg infect dis", "hansen ia", "parise cm", "foster e", "parasit vectors", "eisen", 
    "trends parasitol", "vienna", "schulze tl", "jordan ra", "charbell", "avian flu", "clostridiales", 
    "lachnospira", "illumina miseq", "multianalyte profiling", "milliplex", "white", "ruminococcacea", 
    "bifidobacterium", "weissella", "prevotella", "rump cycle", "related story", "larchmont", 
    "qfs", "qanoninfluencer", "ispos ispos", "bala", "nc house", "astronauts", "mps", "hidden history", 
    "intro", "av breedenbroek", "trans fatty acids", "stricter", "flour", "nuts", "seeds", "frozen", "level", 
    "flu case reported", "hepatocellular carcinoma", "subvariant", "sargassum surge", "rotavac", "latino", 
    "malayalis", "also", "compromise", "buckminster fullter", "togo", "random", "cigarette sales", 
    "townsville", "labels", "front door left open", "epidemic season", "appendix", "author", "caceres", 
    "forbes", "flu assay", "alan price", "baltimor", "consumer dashboards", "calm", "xylitol", "butterfly", 
    "collective", "honourable", "aussie", "credit", "allegiant", "rivac", "enabling kit", "jingle", 
    "countdown", "advance careplex", "southeast raleigh promise", "prem", "nook", "beyfortus", 
    "toronto campus", "thwaites", "follow ap", "cigarette", "cigar smokers", "cigarette smokers", 
    "cigarette smoking", "key findings", "green services", "portland", "insights", "mmr", "sunshine", 
    "source aligning across parkinson", "showing widespread support", "chihuahua", "discusses annovis", 
    "multiorgan", "focal", "assert", "editorial", "neurological disorders", "rett syndrome", 
    "complex regional pain", "flu czar", "van outreach", "slow drains", "sewage backups", "lush grass", 
    "windy hill", "jonestown roat", "main street", "lancaster", "park quarryville", "angie casa de la", 
    "conspiracy", "west climber", "conrad hotel", "marriott residence inn", "crain", "residence inn", 
    "nails survivors exam", "size", "breach", "seen", "rite aid", "kidney donors risk death", 
    "houston methodist", "springville", "st lucia", "tips", "paid programming", "ultimas", "blatter", 
    "brown bottle", "sweet soppressata", "flowers", "beauty everywhere", "yaya", "beira kale", "brassicas", 
    "garden kneeler", "grow ease seed starter", "sandra arévalo", "dirt", "primorac", "blog", 
    "related story", "vaccinia immune globulin intravenous", "kiev", "evenmp", "dashboard creation", 
    "god", "babylon", "sabbath", "biblical", "xpert", "generative ai", "ai", "buttercup cir", "loaves", 
    "linked", "thrill seeker", "dana farber", "harvard cancer", "mmwr morb mortal wkly", 
    "southeast aurora", "growing herpes", "indian embassy", "leqembi", "alzheimera", "babesiosis burden", 
    "nuzyra", "seeing", "philippineshas", "significant drop", "sunshiny", "sizzle", "perform", "status", 
    "measle", "adderall", "virgie", "virgi", "rgs", "chopper", "genitle", "ridge", "head start", "head", 
    "respond", "navigation apps", "thomas kh", "dalili mn", "global stakegholders", "whitehouse", 
    "congressman", "whitehouse office", "gene", "prevention shine", "fox", "son", "regulatory framework", 
    "medetomidine", "zonnic", "morningstar", "barchart", "councilor", "momma", "apena", "morristown", 
    "serologic", "jugular", "ticks", "recursos cinegéticos", "c herraiz", "hyalomma", "jugular vein", 
    "risk", "segura", "charmin ultra", "charmin ultra strong", "served", "heaad start", "contest", 
    "dairy cattle", "backyard", "veggie restaurant", "conscience vc", "boost vc", "copyright businesss", 
    "sugar twin", "highway patrol mandan police", "rheumatic heart", "wyber", "young", "seltzer", 
    "r.m characterizing", "adolesc", "reproductive", "gt", "pure aloe", "veggie van", "guthrie", "pools", 
    "chat gpt", "arafat", "statim", "ultrasound", "cox", "dedham", "avian flu dairy", "ortho oncologist", 
    "consultant neurosurgeon", "terminix", "reduce", "indeed", "nature", "mortality rates", 
    "infectionm rates", "nationally", "lending trr", "sandy springs", "jynneos", "nasal discomfort", 
    "doge", "southington", "intravacc intravacc", "primrose", "align", "purdue", "mental", "basil", 
    "newwspaper", "transport", "w$w", "keys", "w~", "cd", "creative biogene launches", "creative biogene", 
    "creative biogene address", "shirley", "parallel", "baltimore", "fuck", "bitch", "physical", 
    "flu detected", "kent", "calc", "apart", "hz", "sas", "speaks",
    # From scrape of final_dataset_with_attribution.parquet
    "mosquito", "fentanyl", "listeria", "mpox", "marijuana", "bird", "top", "clade", "email", "founder",
    "award", "sci", "tel", "coom", "gracia", "escape", "photo", "fahrenheit", "children", "lassa",
    "trader joe", "khan younis", "malawi malawi", "fig download", "supplementary fig", "bird flu",
    "breakthrough", "issuewire", "email alerts", "iron lung", "measles rubella", "investor relations",
    "tonix medicines", "glaxosmithkline", "sanofi", "novo nordisk", "meltwater", "kantar", "refinitiv",
    "westlaw", "plos", "goma", "rafah", "tamil nadu", "aurora", "j clin", "j nadine", "jp nadda",
    "supplementary figs", "figures", "figure", "discusses breakthrough", "wen email", "email address",
    "breakthrough designations", "malawi nepal", "computing breakthrough", "twitter email", "email print",
    "email feldman", "measles worldwide", "medtech breakthrough", "tech breakthrough", "email pinterest",
    "lake malawi", "email required", "via email", "email andrea", "email ross", "worldwide trends",
    "breakthrough announced",     "worldwide wcp", "khan youniswriting", "worldwide amid", "meager worldwide",
    "v worldwide", "email emery", "email pfeil", "worldwide wcp",
    # User-requested additions
    "overhaul", "mercedes ben", "mercedes benz", "bug forecast", "skincare patrol", "appendix fig",
    "antenna", "genie", "initiated", "prep",
    # From second scrape of final_dataset_with_attribution.parquet
    "fig", "figs", "appendix table", "supplementary material", "appendix similar", "intradermal delivery",
    "appendix c", "supplementary methods", "flu fighters", "trader salmonella", "malawi paxmedica",
    "tabbal relationship", "asami relationship", "supplementary videos", "supplementary note", "appendix b",
    "breakthrough designation", "appendix download", "relief worldwide", "breakthrough hits", "fighting malaria",
    "malawi burundi", "rezazadeh relationship", "worldwide wcbs", "volunteer firefighters", "kyoto worldwide",
    "jeffrey email", "westwicke email", "shrimpton worldwide",
    # From third scrape of final_dataset_with_attribution.parquet
    "worldwide", "malawi", "download", "sanofi pasteur", "sanofi msd", "sanofi gsk", "sanofi industrie",
    "plos pathog", "plos biol", "plos glob", "plos one", "plos biology", "plos trop", "plos pathogens",
    "aurora reservoir", "aurora deals", "aurora borealis", "aurora calling", "aurora democrat", "aurora ct",
    "aurora move", "aurora west", "aurora rsvp", "aurora catholic",
    # From final scrape of final_dataset_with_attribution.parquet
    "follow", "impfallianz gavi", "follow stateline", "follow visby", "follow aveva", "follow walker"
}

# Courtesy/political titles to strip when normalizing names (lowercased)
TITLE_PREFIXES = (
    "dr", "mr", "mrs", "ms", "prof", "professor", "senator", "sen", "rep", "representative",
    "gov", "governor", "president", "pres", "secretary", "sec", "sir", "dame", "lady", "lord",
    "mayor", "minister", "ambassador", "hon", "honorable"
)


def _looks_like_url_or_code(s: str) -> bool:
    """Check if string looks like a URL or code."""
    s2 = s.strip().lower()
    return s2.startswith(("http://","https://")) or any(t in s2 for t in (".com",".org",".gov",".edu",".net","js.id","®"))

def _is_non_person_phrase(name: str, stop: Set[str],
                          derived_phrases: Set[str] | None = None) -> bool:
    """Heuristic: return True if phrase looks like NOT a person name."""
    if not name:
        return True
    s = name.strip()
    s_lower = s.lower()
    if not s:
        return True
    if derived_phrases and s_lower in derived_phrases:
        return True
    if s_lower in EXPLICIT_EXCLUDES:
        return True
    # direct membership
    if s_lower in stop:
        return True
    
    # Single letter tokens (very likely not a person)
    if len(s.strip()) == 1 and s.strip().isalpha():
        return True
    
    # Journal citation patterns like "J. Clin", "A. Al" (journal citations, not names)
    if re.match(r"^[A-Za-z]\.\s+[A-Z]", s) and len(s) < 10:
        return True
    if re.match(r"^[A-Za-z]\.\s+[A-Z][a-z]+\.?$", s) and len(s) < 15:
        return True
    
    # URL/code detection
    if _looks_like_url_or_code(s):
        return True
    
    # Acronym detection: all caps, >=3 chars, no lowercase
    if len(s) >= 3 and s.upper() == s and not re.search(r"[a-z]", s):
        return True
    
    # Bullet/arrow character detection
    if re.match(r"^[•‐\-—–↑↓→←±✓✕✖️\u2000-\u206F\u2E00-\u2E7F\s]+$", s):
        return True
    
    # Digits: tokens with >=2 digits
    if sum(c.isdigit() for c in s) >= 2:
        return True
    
    # Place names starting with "Al-" prefix (common Arabic place prefix)
    if re.match(r"^al-", s_lower) and len(s.split()) == 1:
        return True
    
    # Filter URLs/domains more aggressively
    if re.search(r"\.(com|org|gov|edu|net|php)$", s_lower):
        return True
    
    words = [w for w in s_lower.replace("/", " ").split() if w]
    # any token in stop/locations
    if any(w in stop for w in words):
        return True
    if any(w in US_STATES for w in words) or s_lower in US_STATES:
        return True
    if any(w in COMMON_COUNTRIES for w in words) or s_lower in COMMON_COUNTRIES:
        return True
    # foods / beverages
    if any(w in FOOD_KEYWORDS for w in words) or any(w in BEVERAGE_KEYWORDS for w in words):
        return True
    # company/brand organizational suffixes
    if any(w.strip(".,") in BRAND_SUFFIXES for w in words):
        return True
    # drug-like tokens
    for w in words:
        wl = w.strip(".,").lower()
        if len(wl) >= 5 and any(wl.endswith(suf) for suf in DRUG_SUFFIXES):
            return True
    # suffix-based location hint
    for suf in LOCATION_SUFFIXES:
        if s_lower.endswith(suf):
            return True
    # substring indicators
    for sub in EXCLUSION_SUBSTRINGS:
        if sub in s_lower:
            return True
    # animal-specific phrase e.g., "bald eagles"
    if "eagle" in words or "eagles" in words:
        return True
    # looks like product/brand/medical terms
    if any(k in s_lower for k in ["covid", "coronavirus", "vaccine", "advil", "ibuprofen", "tylenol", "acetaminophen"]):
        return True
    # legislative bills like "Bill 123"
    if re.search(r"\bbill\s+\d+\b", s_lower):
        return True
    return False


def _strip_trailing_exclusion_words(name: str, stop: Set[str]) -> str:
    """
    Remove exclusion words/phrases that appear after a person's name.
    Example: "trump speaks" -> "trump", "biden nasal sprays" -> "biden"
    Handles both single words and multi-word phrases from EXPLICIT_EXCLUDES.
    """
    if not name:
        return name
    parts = [p.strip() for p in name.split() if p.strip()]
    if len(parts) < 2:
        return name
    
    name_lower = name.lower()
    
    # First, check for multi-word exclusion phrases at the end
    # Sort by length (longest first) to match longer phrases first
    multi_word_excludes = sorted([e for e in EXPLICIT_EXCLUDES if " " in e], key=len, reverse=True)
    for phrase in multi_word_excludes:
        # Check if the name ends with this exclusion phrase
        if name_lower.endswith(" " + phrase) or name_lower.endswith(phrase):
            # Find where the phrase starts
            phrase_parts = phrase.split()
            if len(parts) >= len(phrase_parts):
                # Check if the last N tokens match the phrase
                last_tokens = " ".join(parts[-len(phrase_parts):]).lower()
                if last_tokens == phrase:
                    # Remove the phrase
                    parts = parts[:-len(phrase_parts)]
                    if not parts:
                        return name  # Don't remove everything
                    name = " ".join(parts)
                    name_lower = name.lower()
                    break
    
    # Now work backwards token by token, removing single-word exclusions
    # Keep at least the first token (first name)
    result_parts = []
    i = len(parts) - 1
    while i >= 0:
        token_lower = parts[i].lower().strip(".,'\"")
        # Check if this token is an exclusion word
        if token_lower in stop or token_lower in EXPLICIT_EXCLUDES:
            # It's an exclusion word, skip it
            i -= 1
            continue
        # Not an exclusion word, keep everything from start to here
        result_parts = parts[:i+1]
        break
    
    # If we removed everything, keep at least the first token
    if not result_parts and parts:
        result_parts = [parts[0]]
    
    return " ".join(result_parts) if result_parts else name


def _name_core(s: str) -> str:
    """
    Reduce a raw token to 'First Last' (or single token). Safe for empties.
    Case/possessive-insensitive: strip possessives before normalization.
    This is from attribution.py - reduces names to just first and last token.
    """
    if not s or not str(s).strip():
        return ""
    # Strip possessives
    s = re.sub(r"[’']s\b", "", str(s), flags=re.IGNORECASE).strip()
    s = re.sub(r"'s\b", "", s, flags=re.IGNORECASE).strip()
    # Strip titles
    s = re.sub(r'^(dr\.?|mr\.?|mrs\.?|ms\.?|prof\.?|president|pres\.?|senator|sen\.?|rep\.?|representative|gov\.?|governor)\s+', '', s, flags=re.IGNORECASE).strip()
    # Strip suffixes
    s = re.sub(r'\s+(jr\.?|sr\.?|phd|md|rn|esq\.?)$', '', s, flags=re.IGNORECASE).strip()
    # Remove inner punctuation (keep hyphens and apostrophes for names)
    s = re.sub(r"[^\w\s\-\.'']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return ""
    # Remove isolated middle initials (A.) → A
    s = re.sub(r"\b([A-Za-z])\.\b", r"\1", s)
    parts = s.split()
    if not parts:
        return ""
    # Keep only alphabetic-ish tokens or hyphenated names
    parts = [p for p in parts if re.fullmatch(r"[A-Za-z][A-Za-z\-'.']*", p)]
    if not parts:
        return ""
    if len(parts) >= 2:
        return f"{parts[0]} {parts[-1]}"  # Just first and last
    return parts[0]

def _canonicalize_name(name: str) -> str:
    """
    Normalize a name to a canonical form (case-insensitive).
    Maps specific surnames and case variations to canonical full names:
    - "trump" -> "Donald Trump"
    - "biden" -> "Joe Biden"
    - "kennedy" -> "Robert F Kennedy"
    - "harris" -> "Kamala Harris"
    
    For other names, converts to Title Case for full names.
    """
    if not name or not str(name).strip():
        return ""
    
    name_lower = name.strip().lower()
    
    # Specific surname mappings (from attribution.py)
    if name_lower == "trump":
        return "Donald Trump"
    elif name_lower == "biden":
        return "Joe Biden"
    elif name_lower == "kennedy":
        return "Robert F Kennedy"
    elif name_lower == "harris":
        return "Kamala Harris"
    
    # For full names, normalize to Title Case
    parts = name.strip().split()
    if len(parts) >= 2:
        # Title case each part (handle hyphenated names)
        title_parts = []
        for part in parts:
            if '-' in part:
                # Handle hyphenated names like "Mary-Jane"
                hyphen_parts = [p.capitalize() for p in part.split('-')]
                title_parts.append('-'.join(hyphen_parts))
            else:
                title_parts.append(part.capitalize())
        return " ".join(title_parts)
    elif len(parts) == 1:
        # Single token: capitalize first letter
        return parts[0].capitalize()
    
    return name.strip()

def _looks_like_person_full_name(name: str) -> bool:
    """
    Heuristic positive check: two or more tokens that look like names.
    Accepts initials (e.g., 'D.'), hyphenated surnames, apostrophes.
    """
    if not name:
        return False
    parts = [p for p in name.strip().split() if p]
    if len(parts) < 2:
        return False
    def is_name_token(tok: str) -> bool:
        t = tok.strip(".,''")
        if not t:
            return False
        # initial or initial with dot
        if len(t) == 1 and t.isalpha():
            return True
        if len(t) == 2 and t[0].isalpha() and t[1] == ".":
            return True
        # Capitalized or hyphenated capitalized
        if "-" in t:
            subs = t.split("-")
            return all(sub and sub[0].isupper() and (len(sub) == 1 or sub[1:].islower()) for sub in subs)
        return t[0].isupper() and (len(t) == 1 or t[1:].islower())
    # consider first and last tokens primarily
    return is_name_token(parts[0]) and is_name_token(parts[-1])


def _clean_people_cell(cell: object, stop: Set[str]) -> Optional[str]:
    """Clean one cell of comma-separated names using your helpers + stopword filter."""
    if cell is None:
        return None
    s = str(cell)
    if not s.strip():
        return None

    # FIRST: Remove possessives ('s) from names BEFORE processing
    # Pattern: "Joe Biden's" -> "Joe Biden", "Hamilton Hollande's" -> "Hamilton Hollande"
    s = re.sub(r"[’']s\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"'s\b", "", s, flags=re.IGNORECASE)
    
    # SECOND: Remove "Bill" + number patterns (legislative bills, not people)
    # Pattern: "Bill 391", "Bill 390", etc.
    s = re.sub(r'\bBill\s+\d+\b', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\bbill\s+\d+\b', '', s)
    # Clean up extra commas left behind
    s = re.sub(r',\s*,+', ',', s)  # Remove double/triple commas
    s = re.sub(r'^\s*,\s*', '', s)  # Remove leading comma
    s = re.sub(r'\s*,\s*$', '', s)  # Remove trailing comma
    s = s.strip()
    if not s:
        return None

    # Prefer your robust extractors if available
    names: list[str] = []
    if extract_clean_names is not None:
        names = extract_clean_names(s) or []
        # Additional strict validation
        if is_likely_person_name is not None:
            names = [n for n in names if is_likely_person_name(n)]
    else:
        # Simple fallback: split commas and trim
        names = [part.strip() for part in s.split(",") if part.strip()]

    if not names:
        return None

    # Remove names composed only of stopwords; and drop tokens that are stopwords
    cleaned = []
    for n in names:
        # Remove possessives from individual names too (in case they slipped through)
        n = re.sub(r"[’']s\b", "", str(n), flags=re.IGNORECASE).strip()
        n = re.sub(r"'s\b", "", n, flags=re.IGNORECASE).strip()
        if not n:
            continue
        # Skip "Bill" + number patterns that might have slipped through
        if re.search(r'\bbill\s+\d+\b', n, re.IGNORECASE):
            continue
        
        # Strip trailing exclusion words (e.g., "trump speaks" -> "trump")
        n = _strip_trailing_exclusion_words(n, stop)
        if not n:
            continue
        
        # Apply _name_core() to reduce to "First Last" format (from attribution.py)
        n_core = _name_core(n)
        if not n_core:
            continue
        
        # Check if core name is a non-person token (using attribution.py patterns)
        if _is_non_person_phrase(n_core, stop):
            continue
        
        # Canonicalize the name (applies specific mappings and Title Case)
        candidate = _canonicalize_name(n_core)
        if not candidate:
            continue
        
        # Final check: ensure canonicalized name is still valid
        if candidate.lower() in stop or candidate.lower() in EXPLICIT_EXCLUDES:
            continue
        
        # Final non-person phrase check on canonicalized name
        keep = not _is_non_person_phrase(candidate, stop)
        # Guardrail: if this looks like a real full person name, keep it unless clearly organization/location
        if not keep:
            if _looks_like_person_full_name(candidate):
                last = candidate.split()[-1].lower().strip(".,")
                suspicious_last = (
                    last in BRAND_SUFFIXES
                    or last in US_STATES
                    or candidate.lower() in COMMON_COUNTRIES
                    or last in {"university","college","center","centre","hospital","hall","lab","labs","institute"}
                )
                if not suspicious_last:
                    keep = True
        if keep:
            cleaned.append(candidate)

    # Deduplicate but preserve order
    seen = set()
    uniq = []
    for n in cleaned:
        key = n.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(n)

    return (", ".join(uniq)) if uniq else None


def clean_series(series: pd.Series, stop: Set[str]) -> pd.Series:
    return series.fillna("").map(lambda x: _clean_people_cell(x, stop)).astype("string")


def _iter_names(series: pd.Series) -> list[str]:
    """Flatten comma-separated names from a cleaned series into a list."""
    out: list[str] = []
    for cell in series.dropna():
        s = str(cell).strip()
        if not s:
            continue
        parts = [p.strip() for p in s.split(",") if p.strip()]
        out.extend(parts)
    return out


def _build_surname_upgrade_map(series_list: list[pd.Series],
                               min_full_count: int = 3,
                               dominance_ratio: float = 0.6) -> dict[str, str]:
    """
    Build a map from single-token surname -> dominant full name using dataset-wide evidence.
    Only create a mapping if a full name with that surname occurs at least `min_full_count`
    and represents at least `dominance_ratio` of all full names with that surname.
    """
    # Count full names per surname
    surname_to_full_counts: dict[str, dict[str, int]] = {}
    for series in series_list:
        for name in _iter_names(series):
            toks = name.split()
            if len(toks) >= 2:
                surname = toks[-1].lower()
                d = surname_to_full_counts.setdefault(surname, {})
                d[name] = d.get(name, 0) + 1

    upgrade_map: dict[str, str] = {}
    for surname, full_counts in surname_to_full_counts.items():
        total = sum(full_counts.values())
        if total == 0:
            continue
        # dominant full name
        dominant_full, dom_count = max(full_counts.items(), key=lambda kv: kv[1])
        if dom_count >= min_full_count and (dom_count / total) >= dominance_ratio:
            upgrade_map[surname] = dominant_full
    return upgrade_map


def _apply_surname_upgrade(series: pd.Series, upgrade_map: dict[str, str]) -> pd.Series:
    """
    Replace single-token names with the dominant full name for that surname when unambiguous.
    Also upgrade initial + surname patterns (e.g., "D. Trump" -> "Donald Trump")
    Keeps original if no mapping found.
    """
    def upgrade_cell(cell: object) -> Optional[str]:
        if cell is None:
            return None
        s = str(cell).strip()
        if not s:
            return None
        parts = [p.strip() for p in s.split(",") if p.strip()]
        out: list[str] = []
        seen: set[str] = set()
        for name in parts:
            toks = name.split()
            new_name = name
            if len(toks) == 1:
                surname = toks[0].lower()
                mapped = upgrade_map.get(surname)
                if mapped:
                    new_name = mapped
            elif len(toks) == 2:
                first, last = toks[0], toks[1]
                # detect initial formats: "D" or "D." (case-insensitive)
                first_clean = first.replace(".", "").strip()
                if len(first_clean) == 1:
                    surname = last.lower()
                    mapped = upgrade_map.get(surname)
                    if mapped:
                        # mapped like "Donald Trump" -> ensure initial matches
                        mapped_first = mapped.split()[0]
                        if mapped_first and mapped_first[0].lower() == first_clean.lower():
                            new_name = mapped
            key = new_name.lower()
            if key not in seen:
                seen.add(key)
                out.append(new_name)
        return ", ".join(out) if out else None
    return series.map(upgrade_cell).astype("string")


def _merge_within_cell_variants(series: pd.Series, upgrade_map: dict[str, str]) -> pd.Series:
    """
    Within a single cell (comma-separated names), if a full name is present and
    the same row also includes the surname alone, the first name alone, or an initial+surname,
    collapse them to the full name once.
    """
    def _strip_title(name: str) -> str:
        # Remove common titles at the start (with or without dot)
        s = name.strip()
        parts = s.split()
        if not parts:
            return s
        first = parts[0].lower().rstrip(".")
        if first in TITLE_PREFIXES:
            return " ".join(parts[1:]) if len(parts) > 1 else ""
        return s
    def _strip_possessive(name: str) -> str:
        # Drop trailing 's or 's (e.g., "Joe Biden's" -> "Joe Biden", "Hamilton Hollande's" -> "Hamilton Hollande")
        # Handle both straight and curly apostrophes
        name = re.sub(r"[’']s\b", "", name, flags=re.IGNORECASE)
        name = re.sub(r"'s\b", "", name, flags=re.IGNORECASE)
        return name.strip()
    def _strip_suffixes(name: str) -> str:
        # Remove common generational suffixes and punctuation at end
        parts = [p for p in name.strip().split() if p]
        if not parts:
            return name
        SUFFIXES = {"jr","jr.","sr","sr.","iii","ii","iv","v"}
        if parts and parts[-1].lower().rstrip(".") in SUFFIXES:
            parts = parts[:-1]
        return " ".join(parts)
    def _normalize_special_cases(name: str) -> str:
        """
        Map known variants:
        - Robert F Kennedy Jr. -> Robert F Kennedy
        - Robert Kennedy -> Robert F Kennedy (if intent is RFK)
        """
        s = name.strip()
        low = s.lower()
        # Normalize punctuation/spacing
        s = _strip_suffixes(_strip_possessive(s))
        # RFK mappings
        if re.fullmatch(r"robert\s+f\.?\s+kennedy", s, flags=re.IGNORECASE):
            return "Robert F Kennedy"
        if re.fullmatch(r"robert\s+kennedy", s, flags=re.IGNORECASE):
            return "Robert F Kennedy"
        if re.fullmatch(r"robert\s+f\.?\s+kennedy\s+jr\.?", low, flags=re.IGNORECASE):
            return "Robert F Kennedy"
        return s

    def merge_cell(cell: object) -> Optional[str]:
        if cell is None:
            return None
        s = str(cell).strip()
        if not s:
            return None
        parts = [_normalize_special_cases(_strip_possessive(_strip_title(p.strip()))) for p in s.split(",") if p.strip()]
        if not parts:
            return None
        # Collect full names present in this cell
        fulls = [p for p in parts if len(p.split()) >= 2]
        full_last_map: Dict[str, Set[str]] = {}
        full_first_map: Dict[str, Set[str]] = {}
        for fn in fulls:
            toks = fn.split()
            first, last = toks[0], toks[-1]
            full_last_map.setdefault(last.lower(), set()).add(fn)
            full_first_map.setdefault(first.lower(), set()).add(fn)

        out: list[str] = []
        seen: set[str] = set()
        for name in parts:
            toks = name.split()
            new_name = name
            if len(toks) >= 2:
                # initial + surname upgrade inside the row
                first, last = toks[0], toks[-1]
                fc = first.replace(".", "").strip()
                if len(fc) == 1 and last.lower() in full_last_map:
                    candidates = [fn for fn in full_last_map[last.lower()] if fn.split()[0][0].lower() == fc.lower()]
                    if len(candidates) == 1:
                        new_name = candidates[0]
            elif len(toks) == 1:
                tok = toks[0]
                # Single surname -> map to the one full name in row if unique
                if tok.lower() in full_last_map and len(full_last_map[tok.lower()]) == 1:
                    new_name = list(full_last_map[tok.lower()])[0]
                # Single first name -> map if unique in row
                elif tok.lower() in full_first_map and len(full_first_map[tok.lower()]) == 1:
                    new_name = list(full_first_map[tok.lower()])[0]
                else:
                    # As a fallback, use dataset upgrade_map if available
                    mapped = upgrade_map.get(tok.lower())
                    if mapped:
                        new_name = mapped
            key = new_name.lower()
            if key not in seen:
                seen.add(key)
                out.append(new_name)
        return ", ".join(out) if out else None
    return series.map(merge_cell).astype("string")


def run(sampled_path: Path, outdir: Path) -> None:
    """
    Clean person-name columns ONLY in processed_with_people_emotion.parquet.
    Outputs cleaned_sampled_data.parquet to final_data folder.
    """
    stop = _load_stopwords()
    outdir.mkdir(parents=True, exist_ok=True)

    if not sampled_path.exists():
        print(f"[ERROR] processed_with_people_emotion.parquet not found at {sampled_path}")
        print(f"Please ensure the file exists before running.")
        return

    try:
        sampled = pd.read_parquet(sampled_path)
        print(f"[load] Loaded {len(sampled):,} rows from {sampled_path.name}")
        
        # Determine people column
        col = None
        for c in ["people_by_row", "persons_by_row", "persons", "people"]:
            if c in sampled.columns:
                col = c
                break
        
        if col is None:
            print(f"[ERROR] No people column found. Available columns: {list(sampled.columns)}")
            return
        
        print(f"[clean] Cleaning column: {col}")
        col_clean = f"{col}_clean"
        
        # Step 1: Initial cleaning
        sampled[col_clean] = clean_series(sampled[col], stop)
        
        # Step 2: Build surname upgrade map from cleaned data
        print(f"[upgrade] Building surname upgrade map...")
        up_map = _build_surname_upgrade_map([sampled[col_clean]])
        print(f"[upgrade] Found {len(up_map)} surname mappings")
        
        # Step 3: Apply surname upgrades (e.g., "Obama" -> "Barack Obama")
        sampled[col_clean] = _apply_surname_upgrade(sampled[col_clean], up_map)
        
        # Step 4: Merge within-row variants (e.g., "Joe Biden" + "Joe Biden's" -> "Joe Biden")
        print(f"[merge] Merging within-row name variants...")
        sampled[col_clean] = _merge_within_cell_variants(sampled[col_clean], up_map)
        
        # Step 5: Final strict filter - drop any remaining non-person phrases
        print(f"[filter] Applying final non-person filter...")
        def _final_filter_cell(cell: object) -> Optional[str]:
            if cell is None:
                return None
            s = str(cell).strip()
            if not s:
                return None
            keep = []
            for p in [pp.strip() for pp in s.split(",") if pp.strip()]:
                if not _is_non_person_phrase(p, stop):
                    keep.append(p)
            return ", ".join(keep) if keep else None
        
        sampled[col_clean] = sampled[col_clean].map(_final_filter_cell).astype("string")
        
        # Step 6: Drop raw text columns to reduce file size
        print(f"[drop] Dropping raw text columns...")
        columns_to_drop = []
        for col_name in ["article_body_raw", "headline_raw"]:
            if col_name in sampled.columns:
                columns_to_drop.append(col_name)
        if columns_to_drop:
            sampled = sampled.drop(columns=columns_to_drop)
            print(f"[drop] Dropped columns: {columns_to_drop}")
        
        # Write output
        out_path = outdir / "cleaned_sampled_data.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sampled.to_parquet(out_path, index=False)
        print(f"[ok] Wrote cleaned sampled dataset -> {out_path}")
        print(f"[ok] Output shape: {sampled.shape}")
        
    except Exception as e:
        print(f"[ERROR] Failed to process sampled dataset: {e}")
        import traceback
        traceback.print_exc()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Clean person-name columns in processed_with_people_emotion.parquet ONLY.")
    ap.add_argument("--sampled", default=str(ROOT / "data_storage" / "processed_data" / "processed_with_people_emotion.parquet"),
                    help="Path to processed_data/processed_with_people_emotion.parquet (REQUIRED)")
    ap.add_argument("--outdir", default=str(ROOT / "data_storage" / "final_data"),
                    help="Output directory for cleaned_sampled_data.parquet (default: final_data)")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(sampled_path=Path(args.sampled),
        outdir=Path(args.outdir))


