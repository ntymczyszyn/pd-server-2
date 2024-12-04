import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations

# Funkcja do przygotowania danych TF-IDF
def setup_tfidf(data_path):
    df = pd.read_csv(data_path)
    df["NERText"] = df["NER"].apply(lambda x: " ".join(json.loads(x)))
    df["NERList"] = df["NER"].apply(json.loads)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["NERText"])
    print("[TF-IDF] Setup zakończony.")
    return df, vectorizer, tfidf_matrix

# Funkcja do wyszukiwania przepisów
def find_recipe_tfidf(search_terms, df, vectorizer, tfidf_matrix, min_recipes=100):
    results = pd.DataFrame()

    for r in range(len(search_terms), 0, -1):
        term_combinations = list(combinations(search_terms, r))
        temp_results = pd.DataFrame()

        for combination in term_combinations:
            query = " ".join(combination)
            query_vector = vectorizer.transform([query])
            cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

            filtered_indices = cosine_similarities >= 0.1
            filtered_df = df.loc[filtered_indices].copy()

            def full_match(ner_list, combination):
                return all(any(term.lower() in ingredient.lower() for ingredient in ner_list) for term in combination)

            filtered_df["MatchedCount"] = filtered_df["NERList"].apply(
                lambda ner_list: sum(1 for term in combination if any(term.lower() in ingredient.lower() for ingredient in ner_list))
            )

            filtered_df["FullMatch"] = filtered_df["NERList"].apply(lambda ner_list: full_match(ner_list, combination))
            full_match_df = filtered_df[filtered_df["FullMatch"]]

            full_match_df = full_match_df.assign(
                IngredientCount=full_match_df["NERList"].apply(len)
            )
            temp_results = pd.concat([temp_results, full_match_df])

        temp_results = temp_results.sort_values(by="IngredientCount")
        results = pd.concat([results, temp_results])

        if len(results) >= min_recipes:
            break

    unique_results = results.drop_duplicates(subset=["title"]).head(min_recipes)
    return unique_results[["title", "ingredients", "directions", "MatchedCount"]]