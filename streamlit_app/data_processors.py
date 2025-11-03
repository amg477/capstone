"""
Data Processing Functions
Handles data cleaning, transformation, and analysis functions
"""

import streamlit as st
import pandas as pd
import re


def clean_bin_column(df, col_name):
    """Extract just the number from bin columns (e.g., 'circulation_size_bin_5' -> 5)"""
    if col_name not in df.columns:
        return df
    
    df = df.copy()
    
    # Extract number from end of string
    def extract_number(val):
        if pd.isna(val):
            return None
        val_str = str(val)
        # Find last number in the string
        matches = re.findall(r'\d+', val_str)
        if matches:
            return int(matches[-1])  # Get last number
        return None
    
    df[col_name] = df[col_name].apply(extract_number)
    return df


def extract_clean_names(person_string):
    """
    Extract and clean person names from a string that may contain multiple names or non-name words.
    Returns a list of cleaned names.
    """
    if pd.isna(person_string) or not person_string:
        return []
    
    person_str = str(person_string).strip()
    if not person_str:
        return []
    
    # Common non-name words to filter out
    non_name_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
        'he', 'she', 'it', 'we', 'they', 'his', 'her', 'its', 'our', 'their', 'said', 'says', 'say',
        # Common healthcare/policy words that might be mixed in
        'health', 'care', 'policy', 'healthcare', 'government', 'public', 'private', 'state', 'federal', 
        'local', 'national', 'international', 'patient', 'patients', 'doctor', 'doctors', 'medical', 
        'medicine', 'hospital', 'hospitals', 'clinic', 'clinics', 'center', 'centers', 'institute', 
        'institutes', 'university', 'universities', 'college', 'colleges', 'company', 'companies', 
        'organization', 'organizations', 'group', 'groups', 'foundation', 'society', 'association',
        # Common verbs/action words
        'report', 'reports', 'study', 'studies', 'research', 'data', 'information', 'news', 'article', 
        'articles', 'analysis', 'analyses', 'survey', 'surveys', 'poll', 'polls',
        # Time/date words
        'year', 'years', 'time', 'times', 'day', 'days', 'week', 'weeks', 'month', 'months',
        # Location words
        'city', 'cities', 'town', 'towns', 'country', 'countries', 'nation', 'nations',
        # Single letter or very short common words
        'i', 'am', 'go', 'if', 'me', 'my', 'no', 'so', 'up', 'us',
        # Articles and determiners
        'all', 'any', 'each', 'every', 'some', 'both', 'either', 'neither',
        # Other common words
        'new', 'old', 'good', 'bad', 'big', 'small', 'high', 'low', 'first', 'last', 'next', 'previous'
    }
    
    # Split by common delimiters (comma, semicolon, pipe, newline, etc.)
    potential_names = re.split(r'[,;|&\n\r]+', person_str)
    
    cleaned_names = []
    for name_candidate in potential_names:
        name_candidate = name_candidate.strip()
        if not name_candidate:
            continue
        
        # Remove common prefixes/suffixes and clean
        # Remove titles (Dr., Mr., Mrs., Ms., Prof., etc.)
        name_candidate = re.sub(r'^(dr\.?|mr\.?|mrs\.?|ms\.?|prof\.?|professor|senator|rep\.?|representative|gov\.?|governor|president|pres\.?|sen\.?|secretary|sec\.?)\s+', 
                                '', name_candidate, flags=re.IGNORECASE)
        
        # Remove trailing titles/descriptors
        name_candidate = re.sub(r'\s+(jr\.?|sr\.?|phd|md|rn|esq\.?)$', '', name_candidate, flags=re.IGNORECASE)
        
        # Split into words and filter
        words = name_candidate.split()
        filtered_words = []
        for word in words:
            word_clean = word.strip('.,;:!?()[]{}"\'')
            if not word_clean:
                continue
            # Skip if it's a common non-name word (case-insensitive)
            if word_clean.lower() in non_name_words:
                continue
            # Skip if it contains numbers (unlikely to be part of a name)
            if re.search(r'\d', word_clean):
                continue
            # Skip if it's too short (less than 2 characters) unless it's a single initial
            if len(word_clean) < 2 and word_clean.upper() not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                continue
            filtered_words.append(word_clean)
        
        # If we have meaningful words left, it's likely a name
        if filtered_words:
            cleaned_name = ' '.join(filtered_words)
            # Skip if the whole thing is too short (less than 2 characters)
            if len(cleaned_name.replace(' ', '')) >= 2:
                cleaned_names.append(cleaned_name)
    
    return cleaned_names


def normalize_name_for_grouping(name):
    """
    Normalize a name for grouping similar names together.
    Converts to lowercase, removes extra spaces, handles common variations.
    """
    if not name or pd.isna(name):
        return ""
    
    name_str = str(name).strip()
    if not name_str:
        return ""
    
    # Convert to lowercase
    normalized = name_str.lower()
    
    # Remove extra whitespace
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # Remove common prefixes (keep the name part)
    normalized = re.sub(r'^(dr\.?|mr\.?|mrs\.?|ms\.?|prof\.?)\s+', '', normalized)
    
    # Remove trailing suffixes
    normalized = re.sub(r'\s+(jr\.?|sr\.?|phd|md|rn|esq\.?)$', '', normalized)
    
    # Remove punctuation
    normalized = re.sub(r'[^\w\s]', '', normalized)
    
    return normalized.strip()


def is_likely_person_name(name_str):
    """
    Strict check: Does this string look like a person's name?
    Returns False for organizations, abbreviations, email-like patterns, etc.
    Also filters out phrases like "trump epa", "trump jan" that aren't actual names.
    """
    if not name_str or len(name_str.strip()) < 2:
        return False
    
    name_lower = name_str.lower().strip()
    
    # Skip if too short (less than 2 characters)
    if len(name_lower) < 2:
        return False
    
    # Skip if it contains numbers (unlikely to be a name)
    if re.search(r'\d', name_str):
        return False
    
    # Skip if it contains hyphens (likely book titles, URLs, or technical terms)
    if '-' in name_str or '_' in name_str:
        return False
    
    # Skip if it looks like a scientific name (genus/species format like "Lachnobacterium Megasphaera")
    # Typically capitalized first word followed by capitalized second word, both very long
    words = name_str.split()
    if len(words) >= 2:
        # Check if both words are very long and capitalized (typical of scientific names)
        if all(len(w) > 8 and w[0].isupper() for w in words[:2]):
            # Check if it contains typical scientific name patterns
            if any(word.lower() in ['streptococcus', 'haemophilus', 'lachnobacterium', 'megasphaera', 'bacterium', 'bacteria'] for word in words):
                return False
            # Very long words (>10 chars) that are capitalized suggest scientific names
            if len(words[0]) > 10 and len(words[1]) > 10:
                return False
    
    # Skip if it looks like a book/article title (contains "of", "to", "the", "and" in patterns)
    title_patterns = ['-of-', '-to-', '-the-', '-and-', '-an-', '-a-', '-in-', '-on-']
    if any(pattern in name_lower for pattern in title_patterns):
        return False
    
    # Skip if it contains "many", "today", "unhealthy", "foods", "brought" (common in titles)
    title_words = ['many', 'today', 'unhealthy', 'foods', 'brought', 'introduction', 'environmental', 'impact']
    if any(title_word in name_lower for title_word in title_words):
        return False
    
    # Skip if it looks like an email or website
    if '@' in name_str or '.com' in name_lower or '.edu' in name_lower or '.org' in name_lower or '.gov' in name_lower:
        return False
    
    # Skip common organization indicators
    org_indicators = ['inc', 'llc', 'corp', 'ltd', 'co', 'company', 'corporation', 'association', 
                      'society', 'foundation', 'institute', 'university', 'hospital', 'center', 
                      'centre', 'school', 'college', 'academy', 'group', 'organization', 'org',
                      'edu', 'gov', 'com', 'net', 'io', 'epa', 'fda', 'cdc', 'nih', 'who',
                      'acip', 'poe', 'lls', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul',
                      'aug', 'sep', 'oct', 'nov', 'dec']
    if any(f' {ind}' in name_lower or name_lower.endswith(f' {ind}') or f'{ind} ' in name_lower 
           or name_lower.endswith(ind) for ind in org_indicators):
        return False
    
    # Skip if it's all caps abbreviations (like "AAA", "AABP", "AAC")
    if name_str.isupper() and len(name_str) <= 5 and not ' ' in name_str:
        return False
    
    # Get words list (already defined above, but make sure we have it)
    if 'words' not in locals():
        words = name_str.split()
    
    # Skip if it contains too many abbreviations (like "aac djde")
    if len(words) > 1:
        short_words = [w for w in words if len(w) <= 2]
        if len(short_words) > len(words) / 2:  # More than half are very short
            return False
    
    # Filter out phrases where a known name is followed by organization/abbreviation
    # Examples: "trump epa", "trump jan", "biden fda"
    common_first_names = ['trump', 'biden', 'obama', 'clinton', 'bush', 'reagan', 'carter', 'ford']
    for first_name in common_first_names:
        if name_lower.startswith(first_name + ' ') and len(words) > 1:
            # Check if the second word is an abbreviation or org indicator
            second_word = words[1].lower() if len(words) > 1 else ''
            if len(second_word) <= 3 or second_word in org_indicators:
                return False
        # Also check for names stuck together like "trumpfrom"
        if first_name in name_lower and not re.search(rf'\b{first_name}\b', name_lower):
            # Name appears but not as a word boundary (like "trumpfrom")
            return False
    
    # Skip common non-name words
    common_words = {'the', 'and', 'or', 'but', 'for', 'with', 'health', 'care', 'policy', 
                    'government', 'public', 'private', 'state', 'federal', 'local', 'national',
                    'war', 'from', 'to', 'of', 'in', 'on', 'at', 'by'}
    if name_lower in common_words:
        return False
    
    # Skip if it's just initials (like "A A" or "A.B.")
    if len(name_str) <= 4 and re.match(r'^[A-Z]\.?\s?[A-Z]?\.?$', name_str.strip()):
        return False
    
    # Must have at least one letter that's not all the same character
    if len(set(name_lower.replace(' ', ''))) < 2:
        return False
    
    return True


def group_similar_names(name_list):
    """
    Group similar names together - FAST VERSION with STRICT filtering.
    Returns a dictionary mapping canonical names to lists of variants.
    Only includes entries that actually look like person names.
    """
    if not name_list:
        return {}
    
    # Extract names - limit to 2000 entries for speed
    max_entries = 2000
    entries_to_process = name_list[:max_entries] if len(name_list) > max_entries else name_list
    
    all_extracted_names = []
    for name_entry in entries_to_process:
        if pd.isna(name_entry):
            continue
        # Fast extraction - just split by comma and clean basic stuff
        if isinstance(name_entry, str):
            # Simple split by comma
            parts = [p.strip() for p in name_entry.split(',') if p.strip()]
            for part in parts:
                # Quick clean - remove titles
                part_clean = re.sub(r'^(dr\.?|mr\.?|mrs\.?|ms\.?|prof\.?|president|pres\.?|senator|rep\.?)\s+', '', part, flags=re.IGNORECASE).strip()
                
                # STRICT FILTER: Only keep if it looks like a person's name
                if is_likely_person_name(part_clean):
                    all_extracted_names.append(part_clean)
    
    # Get unique names
    unique_names = list(set(all_extracted_names))
    
    # IMPROVED GROUPING: Group similar names, prioritizing proper names
    # Group "Trump", "Donald Trump", "trump" together
    result = {}
    sorted_names = sorted(unique_names, key=lambda x: (len(x.split()), -len(x)), reverse=True)  # Multi-word first, then longest
    
    for name in sorted_names:
        name_lower = name.lower().strip()
        found_group = False
        
        # Extract potential surname/first word for grouping
        name_words = name_lower.split()
        if not name_words:
            continue
        
        # Check if this name should be grouped with existing canonical names
        for canonical in list(result.keys()):
            canonical_lower = canonical.lower()
            canonical_words = canonical_lower.split()
            
            # If one name is a substring of another (e.g., "trump" in "donald trump")
            if len(name_words) == 1 and name_words[0] in canonical_words:
                # Single word name matches a word in canonical (e.g., "trump" matches "donald trump")
                result[canonical].append(name)
                found_group = True
                break
            elif len(canonical_words) == 1 and canonical_words[0] in name_words:
                # Canonical is single word and appears in this name (e.g., "trump" in "donald trump")
                # Replace canonical with the longer name
                old_variants = result.pop(canonical)
                result[name] = old_variants + [name]
                found_group = True
                break
            elif name_lower == canonical_lower:
                # Exact match (case-insensitive)
                if name not in result[canonical]:
                    result[canonical].append(name)
                found_group = True
                break
        
        if not found_group:
            result[name] = [name]
    
    # Final cleanup: Use the longest/most complete name as canonical for each group
    final_result = {}
    for canonical, variants in result.items():
        # Use the longest variant, or if tied, the one with most words
        best_canonical = max(variants, key=lambda x: (len(x.split()), len(x)))
        final_result[best_canonical] = sorted(list(set(variants)))  # Remove duplicates
    
    return final_result


@st.cache_data(ttl=3600, show_spinner=False)
def get_cleaned_person_list(df, person_column='person_list'):
    """
    Get a cleaned and grouped list of person names from a dataframe column.
    Returns a sorted list of canonical names and a mapping for filtering.
    FAST VERSION: Simplified processing for speed.
    CACHED: First run processes, subsequent runs are instant!
    """
    if df is None or df.empty or person_column not in df.columns:
        return [], {}
    
    # Get all unique person_list entries
    unique_entries = df[person_column].dropna().unique().tolist()
    
    # For large datasets, limit processing significantly for speed
    if len(unique_entries) > 2000:
        # Just take first 2000 - they're usually sorted by frequency
        unique_entries = unique_entries[:2000]
    
    # Group similar names (simplified fast version)
    name_groups = group_similar_names(unique_entries)
    
    # Get canonical names (sorted)
    canonical_names = sorted(name_groups.keys())
    
    # Create reverse mapping: any variant -> canonical name (for filtering)
    variant_to_canonical = {}
    for canonical, variants in name_groups.items():
        for variant in variants:
            variant_to_canonical[variant.lower()] = canonical
    
    return canonical_names, variant_to_canonical

