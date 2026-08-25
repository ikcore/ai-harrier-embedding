"""Regenerate the Postman collection and environments in this directory.

    python examples/postman/generate.py

The requests mirror examples/http/*.http one-to-one; edit this file (not the
JSON) when the API or the examples change. Standard library only.
"""
import json, os, uuid

OUT = os.path.dirname(os.path.abspath(__file__))

NS = uuid.NAMESPACE_URL


def uid(name):
    return str(uuid.uuid5(NS, "https://github.com/ikcore/harrier_embeddings/postman/" + name))


def test_script(lines):
    return {"listen": "test", "script": {"type": "text/javascript", "exec": lines}}


def status_test(code):
    return [f'pm.test("status {code}", () => pm.response.to.have.status({code}));']


def vectors_test(count, dims_var="dims"):
    return [
        'const r = pm.response.json();',
        f'pm.test("{count} vector(s) in input order", () => pm.expect(r.output).to.have.length({count}));',
        'pm.test("vectors have the model\'s size", () => {',
        f'  const dims = Number(pm.collectionVariables.get("{dims_var}") || 0);',
        '  if (dims) pm.expect(r.output[0]).to.have.length(dims);',
        '});',
        'pm.test("usage carries the token count", () => pm.expect(r.usage.total.total_tokens).to.be.a("number"));',
    ]


def unit_length_test():
    return [
        'pm.test("vectors are unit length", () => {',
        '  for (const v of pm.response.json().output) {',
        '    const n = Math.sqrt(v.reduce((a, x) => a + x * x, 0));',
        '    pm.expect(Math.abs(n - 1)).to.be.below(1e-3);',
        '  }',
        '});',
    ]


def req(name, method, path, body=None, desc="", tests=None, query=None, raw_body=None, no_content_type=False, status=200):
    url = {"raw": "{{host}}" + path + (("?" + "&".join(f"{k}={v}" for k, v in query)) if query else ""),
           "host": ["{{host}}"],
           "path": [p for p in path.strip("/").split("/") if p]}
    if query:
        url["query"] = [{"key": k, "value": v} for k, v in query]
    r = {"method": method, "header": [], "url": url, "description": desc}
    if body is not None or raw_body is not None:
        if not no_content_type:
            r["header"].append({"key": "Content-Type", "value": "application/json"})
        r["body"] = {"mode": "raw",
                     "raw": raw_body if raw_body is not None else json.dumps(body, indent=2, ensure_ascii=False),
                     "options": {"raw": {"language": "json"}}}
    events = [test_script(status_test(status) + (tests or []))]
    return {"name": name, "request": r, "response": [], "event": events}


def folder(name, items, desc=""):
    return {"name": name, "description": desc, "item": items}


MODEL = "{{model}}"
DOCS = [
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
]

# ---------------------------------------------------------------- health
health = folder("Health", [
    req("Health", "GET", "/health",
        desc="Liveness / readiness: status, version, uptime, loaded models, compiled backends. Also records the default model and its dimensions for the other requests' tests.",
        tests=[
            'const h = pm.response.json();',
            'pm.test("status ok", () => pm.expect(h.status).to.eql("ok"));',
            'pm.test("at least one model loaded", () => pm.expect(h.models.length).to.be.above(0));',
        ]),
    req("Root serves the same payload", "GET", "/",
        desc="`GET /` returns the health document too.",
        tests=['pm.test("same shape as /health", () => pm.expect(pm.response.json().service).to.eql("harrier_api"));']),
], "Liveness / readiness for harrier_api.")

# ---------------------------------------------------------------- models
models = folder("Models", [
    req("All loaded models", "GET", "/v1/models",
        desc="GAISe-shaped model records: id, provider (`harrier`), display_name, status, capabilities, limits, plus an `embedding` profile (dimension_rule: fixed, task_control, resolved task_instructions) and `runtime` facts (architecture, source, device, dtype, quantization, dequantized, load_seconds, default). The test stores the `{{model}}` record's dimensions as `dims` for the embeddings tests.",
        tests=[
            'const ms = pm.response.json().models;',
            'pm.test("at least one record", () => pm.expect(ms.length).to.be.above(0));',
            'pm.test("records carry a runtime block", () => pm.expect(ms[0].runtime).to.have.property("device"));',
            'const want = pm.collectionVariables.get("model").replace(/^.*::/, "");',
            'const m = ms.find(x => x.id.replace(/^.*::/, "") === want) || ms[0];',
            'pm.collectionVariables.set("dims", m.limits.embedding_dimensions);',
            'for (const x of ms) {',
            '  pm.collectionVariables.set("dims_" + x.id.replace(/^.*::/, ""), x.limits.embedding_dimensions);',
            '  if (x.runtime.default) pm.collectionVariables.set("dims_default", x.limits.embedding_dimensions);',
            '}',
            'console.log("model", m.id, "runs on", m.runtime.device, m.runtime.quantization || m.runtime.dtype, "→", m.limits.embedding_dimensions, "dims");',
        ]),
    req("Filter by operation=embeddings", "GET", "/v1/models", query=[("operation", "embeddings")],
        desc="`embeddings` (also `embedding`, `embed`) returns everything.",
        tests=['pm.test("not empty", () => pm.expect(pm.response.json().models.length).to.be.above(0));']),
    req("Filter by operation=instruct (empty)", "GET", "/v1/models", query=[("operation", "instruct")],
        desc="Other GAISe operations are valid but this service offers none → empty list.",
        tests=['pm.test("empty list", () => pm.expect(pm.response.json().models).to.have.length(0));']),
    req("Filter by provider=harrier", "GET", "/v1/models", query=[("provider", "harrier")],
        desc="`harrier` returns everything.",
        tests=['pm.test("not empty", () => pm.expect(pm.response.json().models.length).to.be.above(0));']),
    req("Filter by provider=openai (empty)", "GET", "/v1/models", query=[("provider", "openai")],
        desc="Any other provider → empty list.",
        tests=['pm.test("empty list", () => pm.expect(pm.response.json().models).to.have.length(0));']),
    req("Both filters, as a GAISe router sends them", "GET", "/v1/models",
        query=[("operation", "embeddings"), ("provider", "harrier")],
        tests=['pm.test("not empty", () => pm.expect(pm.response.json().models.length).to.be.above(0));']),
    req("One model by GAISe-qualified id", "GET", "/v1/models/{{model}}",
        desc="`harrier::harrier-oss-v1-0.6b`.",
        tests=['pm.test("single record", () => pm.expect(pm.response.json()).to.have.property("embedding"));']),
    req("One model by bare id", "GET", "/v1/models/harrier-oss-v1-270m",
        desc="Needs the 270m loaded (`HARRIER_MODELS=270m,0.6b`).",
        tests=['pm.test("single record", () => pm.expect(pm.response.json().id).to.eql("harrier::harrier-oss-v1-270m"));']),
    req("One model by alias", "GET", "/v1/models/270m",
        desc="Needs the 270m loaded.",
        tests=['pm.test("single record", () => pm.expect(pm.response.json().id).to.eql("harrier::harrier-oss-v1-270m"));']),
    folder("Errors", [
        req("400 — unknown operation", "GET", "/v1/models", query=[("operation", "bogus")], status=400,
            desc="Plain-text body, like gaise-api."),
        req("404 — model that is not loaded", "GET", "/v1/models/harrier::harrier-oss-v1-27b", status=404),
    ], "Plain-text bodies, like gaise-api."),
], "GET /v1/models — GAISe-shaped model records.")

# ---------------------------------------------------------------- embeddings
def emb(name, body, desc="", tests=None, count=None, unit=True, status=200, dims_var="dims", **kw):
    t = []
    if status == 200:
        n = count if count is not None else (len(body["input"]) if isinstance(body.get("input"), list) else 1)
        t += vectors_test(n, dims_var)
        if unit:
            t += unit_length_test()
    t += tests or []
    return req(name, "POST", "/v1/embeddings", body=body, desc=desc, tests=t, status=status, **kw)


model_card = folder("Model-card example", [
    emb("Minimal — one string, default model, default task", {
        "input": "Definition of summit for English Language Learners: the highest point of a mountain."},
        desc="No `model` → the service's default (first of HARRIER_MODELS); no `task` → `document`.",
        dims_var="dims_default"),
    emb("Queries — task: query", {
        "model": MODEL, "task": "query",
        "input": ["how much protein should a female eat", "summit define"]},
        desc="`query` wraps each input as `Instruct: Given a web search query, retrieve relevant passages that answer the query\\nQuery: {text}`. The test keeps the vectors for the documents request.",
        tests=['pm.collectionVariables.set("query_vectors", JSON.stringify(pm.response.json().output));']),
    emb("Documents — task: document", {
        "model": MODEL, "task": "document", "input": DOCS},
        desc="Documents are embedded bare (no instruction). Dot product of queries × documents reproduces the model card's `[[66.27, 28.46], [29.77, 69.94]]` (×100) on the 0.6b; the test checks it within ±1.5 (bf16 / int8 drift) when `{{model}}` is the 0.6b.",
        tests=[
            'const q = JSON.parse(pm.collectionVariables.get("query_vectors") || "null");',
            'const d = pm.response.json().output;',
            'if (q && /0\\.6b$/.test(pm.collectionVariables.get("model"))) {',
            '  const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0) * 100;',
            '  const scores = q.map(qv => d.map(dv => dot(qv, dv)));',
            '  console.log("scores", JSON.stringify(scores.map(r => r.map(x => +x.toFixed(2)))));',
            '  const want = [[66.27, 28.46], [29.77, 69.94]];',
            '  pm.test("model-card scores reproduce (±1.5)", () => {',
            '    for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) pm.expect(Math.abs(scores[i][j] - want[i][j])).to.be.below(1.5);',
            '  });',
            '}',
        ]),
], "The example from the Harrier model card, query side and document side.")

D270 = "dims_harrier-oss-v1-270m"
names = folder("Model names", [
    emb("Registry alias", {"model": "270m", "input": "alias"}, desc="Needs the 270m loaded.", dims_var=D270),
    emb("Bare registry id", {"model": "harrier-oss-v1-270m", "input": "bare id"}, dims_var=D270),
    emb("Hugging Face repo", {"model": "microsoft/harrier-oss-v1-270m", "input": "hub repo"}, dims_var=D270),
    emb("Foreign provider:: prefix is stripped", {"model": "some-router::harrier-oss-v1-270m", "input": "foreign provider prefix"},
        desc="Any `provider::` prefix is stripped, so a GAISe router can forward requests unchanged.", dims_var=D270),
], "All of these resolve to the same loaded model — the 270m, so run the server with HARRIER_MODELS=270m,0.6b. Their vector-size test uses the 270m's dimensions recorded by the Models request.")

tasks = folder("Tasks", [
    emb("similarity", {"model": MODEL, "task": "similarity",
                       "input": ["The cat sat on the mat.", "A cat is sitting on a mat.", "Interest rates rose again today."]},
        desc="Symmetric: the instruction `Retrieve semantically similar text` goes on every input.",
        tests=[
            'const [a, b, c] = pm.response.json().output;',
            'const dot = (x, y) => x.reduce((s, v, i) => s + v * y[i], 0);',
            'pm.test("paraphrases score higher than the unrelated sentence", () => pm.expect(dot(a, b)).to.be.above(dot(a, c)));',
        ]),
    emb("classification", {"model": MODEL, "task": "classification",
                           "input": ["This product broke after two days, very disappointed.", "Absolutely love it, works perfectly."]},
        desc="`Classify the given text into one of the predefined categories`"),
    emb("clustering", {"model": MODEL, "task": "clustering",
                       "input": ["Quarterly earnings beat expectations", "New vaccine trial shows promise", "Central bank holds rates steady"]},
        desc="`Identify the topic or theme of the given text`"),
    emb("code_query", {"model": MODEL, "task": "code_query", "input": "read a file to a string in rust"},
        desc="`Given a code search query, retrieve relevant code snippets or documentation that answer the query`. Index the code itself with `task: document`."),
    emb("fact_verification", {"model": MODEL, "task": "fact_verification",
                              "input": "The Great Wall of China is visible from the Moon with the naked eye."},
        desc="`Given a claim, retrieve documents that support or refute the claim`"),
    emb("question_answering", {"model": MODEL, "task": "question_answering",
                               "input": "What year did the Apollo 11 mission land on the Moon?"},
        desc="`Given a question, retrieve passages that answer the question`"),
    emb("bitext_query", {"model": MODEL, "task": "bitext_query",
                         "input": ["Where is the railway station?", "Où est la gare ?"]},
        desc="`Retrieve parallel sentences` (cross-lingual mining)",
        tests=[
            'const [en, fr] = pm.response.json().output;',
            'pm.test("translations are close", () => pm.expect(en.reduce((s, v, i) => s + v * fr[i], 0)).to.be.above(0.7));',
        ]),
    emb("Task aliases — case-insensitive, - and space = _", {"model": MODEL, "task": "Search-Query", "input": "aliases work too"}),
    emb("Unknown task = custom instruction (verbatim)", {"model": MODEL,
                                                        "task": "Given a legal question, retrieve the relevant statute",
                                                        "input": "Can a landlord keep the deposit for normal wear and tear?"},
        desc="The escape hatch: anything that is not a known task is used verbatim as the instruction."),
    emb("Explicit instruction overrides task", {"model": MODEL, "task": "query",
                                               "instruction": "Given a product review, retrieve reviews that express the same opinion",
                                               "input": ["Battery life is excellent", "Lasts all day on one charge"]},
        desc="Harrier extension — `instruction` overrides `task` and is applied to every input."),
], "Every canonical GAISe task; see README “Tasks → instructions”.")

other = folder("Other fields", [
    emb("normalize: false — raw pooled hidden state", {"model": MODEL, "input": "raw, not L2-normalised", "normalize": False},
        unit=False,
        tests=[
            'pm.test("not unit length", () => {',
            '  const v = pm.response.json().output[0];',
            '  pm.expect(Math.abs(Math.sqrt(v.reduce((a, x) => a + x * x, 0)) - 1)).to.be.above(1e-3);',
            '});',
        ]),
    emb("dimensions — accepted and ignored", {"model": MODEL, "input": "still comes back at the model's native size", "dimensions": 256},
        desc="Harrier vectors are fixed-size (GAISe's `fixed` rule); the vector-size test confirms the native size."),
    emb("correlation_id — passthrough", {"model": MODEL, "correlation_id": "req-7f3a2c", "task": "query", "input": "tracing passthrough"},
        desc="Accepted, not used; a router's tracing key travels through untouched."),
    emb("Mixed lengths in one request", {"model": MODEL, "input": [
        "short", "",
        "A medium-length sentence that has a few more tokens than the one before it.",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."]},
        desc="Batched by length internally; short rows are never padded to the long one. An empty string is a valid input (a lone EOS token)."),
])

errors = folder("Errors", [
    emb("400 — empty input", {"model": MODEL, "input": []}, status=400,
        tests=['pm.test("plain-text body", () => pm.expect(pm.response.text()).to.include("input"));']),
    req("400 — malformed JSON", "POST", "/v1/embeddings", raw_body='{ "input": "missing closing brace"', status=400),
    emb("404 — model that is not loaded", {"model": "harrier::harrier-oss-v1-27b", "input": "not loaded"}, status=404,
        desc="The body lists what is loaded.",
        tests=['pm.test("body lists the loaded models", () => pm.expect(pm.response.text()).to.include("loaded"));']),
    req("415 — missing Content-Type", "POST", "/v1/embeddings", raw_body='{ "input": "no content type" }',
        no_content_type=True, status=415),
], "Plain-text bodies, like gaise-api.")

embeddings = folder("Embeddings", [model_card, names, tasks, other, errors],
                    "POST /v1/embeddings — the GAISe `GaiseEmbeddingsRequest` contract. Response: `{ output: [[f32; dims], …], usage: { input: { prompt_tokens }, total: { total_tokens } } }`. Vectors come back in input order, unit-length unless `normalize: false`. 270m → 640 dims, 0.6b → 1024 dims.")

collection = {
    "info": {
        "_postman_id": uid("collection"),
        "name": "harrier_api",
        "description": "Microsoft Harrier OSS v1 embeddings served by harrier_api — every endpoint, task, model-name form and error case. Mirrors examples/http/*.http.\n\nStart a server first, e.g. `HARRIER_MODELS=270m,0.6b target/release/harrier_api` (port 3000) or `docker compose up -d` (8-bit GGUF on port 13265), pick the matching environment, then run the collection top to bottom: **Health → Models** records the default model's dimensions, **Embeddings** checks vector count, size and normalisation on every request, and the model-card example verifies the published scores.\n\nThe `Model names` folder and the by-id/alias model requests assume the 270m is loaded too.",
        "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
    },
    "item": [health, models, embeddings],
    "variable": [
        {"key": "host", "value": "http://localhost:3000", "type": "string"},
        {"key": "model", "value": "harrier::harrier-oss-v1-0.6b", "type": "string"},
        {"key": "dims", "value": "", "type": "string"},
        {"key": "query_vectors", "value": "", "type": "string"},
    ],
}


def env(name, host, model="harrier::harrier-oss-v1-0.6b"):
    return {
        "id": uid("env-" + name),
        "name": name,
        "values": [
            {"key": "host", "value": host, "enabled": True},
            {"key": "model", "value": model, "enabled": True},
        ],
        "_postman_variable_scope": "environment",
    }


with open(os.path.join(OUT, "harrier_api.postman_collection.json"), "w", encoding="utf-8", newline="\n") as f:
    json.dump(collection, f, indent=2, ensure_ascii=False)
    f.write("\n")
with open(os.path.join(OUT, "local.postman_environment.json"), "w", encoding="utf-8", newline="\n") as f:
    json.dump(env("harrier_api local (3000)", "http://localhost:3000"), f, indent=2)
    f.write("\n")
with open(os.path.join(OUT, "docker-gguf.postman_environment.json"), "w", encoding="utf-8", newline="\n") as f:
    json.dump(env("harrier_api docker GGUF (13265)", "http://localhost:13265"), f, indent=2)
    f.write("\n")

n = sum(1 for _ in json.dumps(collection).split('"request":')) - 1
print("requests:", n)
