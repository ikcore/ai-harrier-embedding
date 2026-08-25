# Postman collection

`harrier_api.postman_collection.json` — every request from `../http/*.http`
(38 in all: health, models with filters and errors, embeddings for the
model-card example, every model-name form, every task, the other request
fields, and the error cases), each with tests.

1. Import the collection and one environment:
   - `local.postman_environment.json` — `http://localhost:3000` (a native `harrier_api`)
   - `docker-gguf.postman_environment.json` — `http://localhost:13265` (`docker compose up -d`)
2. Start a server with both models so the name/alias examples resolve:
   `HARRIER_MODELS=270m,0.6b target/release/harrier_api` (the GGUF image serves the 0.6B only;
   the `Model names` folder and the by-id/alias model requests then 404 as expected).
3. Run the collection in order. `Models → All loaded models` records each model's
   dimensions; the embeddings tests check vector count, size and unit length, and
   `Documents` recomputes the model card's `[[66.27, 28.46], [29.77, 69.94]]` scores
   from the `Queries` vectors when `model` is the 0.6b.

Variables: `host` and `model` (default `harrier::harrier-oss-v1-0.6b`) come from the
environment; `dims*` and `query_vectors` are set by the tests at run time.

Regenerate after changing the API or the examples with `python generate.py`
(standard library only) rather than editing the JSON by hand.
