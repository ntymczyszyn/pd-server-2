from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import time
from typing import List
from recipe_search import setup_tfidf, find_recipe_tfidf
import json
from contextlib import asynccontextmanager

# global variables
df = None
vectorizer = None
tfidf_matrix = None


class Recipe(BaseModel):
    title: str
    ingredients: List[str]
    instructions: str
    matchedIngredientsCount: int


def process_recipe_results(results):
    processed_recipes = []
    for _, row in results.iterrows():
        title = row["title"]
        ingredients = json.loads(row["ingredients"])
        instructions = " ".join(json.loads(row["directions"]))
        matched_count = row["MatchedCount"]

        recipe = Recipe(
            title=title,
            ingredients=ingredients,
            instructions=instructions,
            matchedIngredientsCount=matched_count,
        )
        processed_recipes.append(recipe)

    return processed_recipes


@asynccontextmanager
async def lifespan(app: FastAPI):
    global df, vectorizer, tfidf_matrix
    data_path = "data/reduced_100000_no_duplicates_filtered__RecipeNLG_dataset.csv"
    print("[Startup] Loading data...")
    df, vectorizer, tfidf_matrix = setup_tfidf(data_path)
    print("[Startup] Data loaded.")
    yield  # Zasoby gotowe, aplikacja działa
    print("[Shutdown] Releasing the resources...")


app = FastAPI(lifespan=lifespan)


@app.get("/recipes/")
async def get_recipes(product: List[str] = Query(...), num_of_recipes: int = 100):
    if df is None or vectorizer is None or tfidf_matrix is None:
        raise HTTPException(status_code=500, detail="Data wa not loaded.")

    recognized_products = product
    start_time = time.time()
    results = find_recipe_tfidf(
        recognized_products, df, vectorizer, tfidf_matrix, num_of_recipes
    )
    end_time = time.time()
    print(
        f"[RecipeSearch] Found {len(results)} recipes in {end_time - start_time:.2f} s."
    )

    recipes = process_recipe_results(results)
    return recipes


@app.get("/")
async def root():
    return {"status": "Server is running"}
