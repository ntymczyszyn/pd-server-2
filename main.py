from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from recipe_search import setup_tfidf, find_recipe_tfidf
import json
from contextlib import asynccontextmanager

# Globalne
df = None
vectorizer = None
tfidf_matrix = None

class RecognizedProduct(BaseModel):
    name: str
    count: int

class Ingredient(BaseModel):
    name: str

class Recipe(BaseModel):
    title: str
    ingredients: List[Ingredient]
    instructions: str
    matchedIngredientsCount: int

    
def process_recipe_results(results):
    processed_recipes = []
    for _, row in results.iterrows():
        title = row["title"]
        raw_ingredients = json.loads(row["ingredients"])
        ingredients = [Ingredient(name=ingredient) for ingredient in raw_ingredients]
        instructions = " ".join(json.loads(row["directions"]))
        matched_count = row["MatchedCount"]

        recipe = Recipe(
            title=title,
            ingredients=ingredients,
            instructions=instructions,
            matchedIngredientsCount=matched_count 
        )
        processed_recipes.append(recipe)
    return processed_recipes

@asynccontextmanager
async def lifespan(app: FastAPI):
    global df, vectorizer, tfidf_matrix
    data_path = "data/reduced_100000_no_duplicates_filtered__RecipeNLG_dataset.csv"
    print("[Startup] Ładowanie danych...")
    df, vectorizer, tfidf_matrix = setup_tfidf(data_path)
    print("[Startup] Dane załadowane.")
    yield  # Zasoby gotowe, aplikacja działa
    print("[Shutdown] Zwalnianie zasobów...")

app = FastAPI(lifespan=lifespan)

@app.post("/recipes/")
async def get_recipes(ingredients: List[RecognizedProduct], min_recipes: int = 100):
    if df is None or vectorizer is None or tfidf_matrix is None:
        raise HTTPException(status_code=500, detail="Dane nie zostały załadowane.")

    search_terms = [product.name for product in ingredients]
    results = find_recipe_tfidf(search_terms, df, vectorizer, tfidf_matrix, min_recipes)
    if results.empty:
        return []

    recipes = process_recipe_results(results)
    print(recipes)
    return recipes

@app.get("/")
async def root():
    print("Serwer działa poprawnie!")  
    return {"status": "Server is running"}

