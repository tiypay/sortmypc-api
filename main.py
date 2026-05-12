import os
import json
import base64
import io
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager

import bcrypt
import jwt
import stripe
import anthropic
import psycopg2
import psycopg2.extras
from PIL import Image
from fastapi import FastAPI, HTTPException, Depends, Header, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
SUPABASE_DB_URL   = os.getenv("SUPABASE_DB_URL", "")
JWT_SECRET        = os.getenv("JWT_SECRET", "change-me")
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PRICE_ID   = os.getenv("STRIPE_PRICE_ID", "")
BACKEND_URL       = os.getenv("BACKEND_URL", "http://localhost:8000")

AI_MODEL       = "claude-haiku-4-5-20251001"
FILES_PER_CREDIT = 100
JWT_EXPIRE_DAYS  = 30
MAX_IMAGE_PX     = 300
MAX_IMAGES_CALL  = 10
BATCH_SIZE       = 100

stripe.api_key = STRIPE_SECRET_KEY

# ── Database ──────────────────────────────────────────────────────────────────

def get_conn():
    return psycopg2.connect(SUPABASE_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)


def init_db():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    credits INTEGER DEFAULT 13,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE TABLE IF NOT EXISTS payments (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID REFERENCES users(id),
                    stripe_session_id TEXT UNIQUE,
                    credits_purchased INTEGER,
                    amount_cents INTEGER,
                    status TEXT DEFAULT 'pending',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
        conn.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if SUPABASE_DB_URL:
        init_db()
    yield

app = FastAPI(title="SortMyPC API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Auth helpers ──────────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


def create_token(user_id: str, email: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(days=JWT_EXPIRE_DAYS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expiré")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Token invalide")


def get_current_user(authorization: str = Header(...)) -> dict:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Format: Bearer <token>")
    payload = decode_token(authorization[7:])
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, email, credits FROM users WHERE id = %s", (payload["sub"],))
            user = cur.fetchone()
    if not user:
        raise HTTPException(status_code=401, detail="Utilisateur introuvable")
    return dict(user)

# ── Schemas ───────────────────────────────────────────────────────────────────

class AuthRequest(BaseModel):
    email: EmailStr
    password: str

class FileItem(BaseModel):
    name: str
    extension: str
    image_b64: str | None = None
    text_preview: str | None = None

class SortRequest(BaseModel):
    files: list[FileItem]

class CheckoutRequest(BaseModel):
    credits: int  # number of credit packs to buy

# ── Auth routes ───────────────────────────────────────────────────────────────

@app.post("/auth/register")
def register(body: AuthRequest):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE email = %s", (body.email,))
            if cur.fetchone():
                raise HTTPException(status_code=400, detail="Email déjà utilisé")
            cur.execute(
                "INSERT INTO users (email, password_hash) VALUES (%s, %s) RETURNING id, credits",
                (body.email, hash_password(body.password)),
            )
            row = cur.fetchone()
        conn.commit()
    token = create_token(str(row["id"]), body.email)
    return {"token": token, "email": body.email, "credits": row["credits"]}


@app.post("/auth/login")
def login(body: AuthRequest):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, password_hash, credits FROM users WHERE email = %s", (body.email,))
            user = cur.fetchone()
    if not user or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Email ou mot de passe incorrect")
    token = create_token(str(user["id"]), body.email)
    return {"token": token, "email": body.email, "credits": user["credits"]}


@app.get("/me")
def me(user: dict = Depends(get_current_user)):
    return {"email": user["email"], "credits": user["credits"]}

# ── Sort route ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "Tu es un expert en organisation de fichiers. "
    "Tu reçois une liste de fichiers avec parfois un aperçu visuel ou textuel. "
    "Propose une structure de dossiers logique. "
    "Réponds UNIQUEMENT en JSON valide. "
    'Format : {"Nom_Dossier": ["fichier1.pdf", "fichier2.jpg"], ...}. '
    "Noms de dossiers courts et clairs en français. Mets les inclassables dans 'Divers'."
)


def _parse_json(raw: str) -> dict:
    import re
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        fixed = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw)
        return json.loads(fixed)


def _sort_batch(client: anthropic.Anthropic, batch: list[FileItem]) -> dict:
    intro_lines = ["Fichiers à organiser :\n"]
    image_blocks = []
    images_used = 0

    for f in batch:
        if f.image_b64 and images_used < MAX_IMAGES_CALL:
            image_blocks.append({"type": "text", "text": f"\n[{f.name}] :"})
            image_blocks.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": f.image_b64},
            })
            images_used += 1
        elif f.text_preview:
            intro_lines.append(f"[{f.name}] → {f.text_preview[:150]}")
        else:
            intro_lines.append(f.name)

    content = [{"type": "text", "text": "\n".join(intro_lines)}] + image_blocks
    msg = client.messages.create(
        model=AI_MODEL, max_tokens=4096, system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )
    return _parse_json(msg.content[0].text)


@app.post("/sort")
def sort_files(body: SortRequest, user: dict = Depends(get_current_user)):
    needed = max(1, -(-len(body.files) // FILES_PER_CREDIT))
    if user["credits"] < needed:
        raise HTTPException(status_code=402, detail=f"Crédits insuffisants ({user['credits']}/{needed})")

    # Debit credits first
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET credits = credits - %s WHERE id = %s AND credits >= %s RETURNING credits",
                (needed, user["id"], needed),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=402, detail="Crédits insuffisants")
        conn.commit()
        remaining = row["credits"]

    try:
        ai_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        batches = [body.files[i:i + BATCH_SIZE] for i in range(0, len(body.files), BATCH_SIZE)]
        merged: dict = {}
        for batch in batches:
            plan = _sort_batch(ai_client, batch)
            for folder, files in plan.items():
                merged.setdefault(folder, []).extend(files)
    except Exception as e:
        # Refund on AI error
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE users SET credits = credits + %s WHERE id = %s", (needed, user["id"]))
            conn.commit()
        raise HTTPException(status_code=500, detail=f"Erreur IA : {e}")

    return {"plan": merged, "credits_used": needed, "credits_remaining": remaining}

# ── Payments ──────────────────────────────────────────────────────────────────

@app.post("/payments/checkout")
def create_checkout(body: CheckoutRequest, user: dict = Depends(get_current_user)):
    if not STRIPE_PRICE_ID:
        raise HTTPException(status_code=500, detail="Stripe non configuré")
    session = stripe.checkout.Session.create(
        payment_method_types=["card"],
        line_items=[{"price": STRIPE_PRICE_ID, "quantity": body.credits}],
        mode="payment",
        success_url=f"{BACKEND_URL}/payments/success?session_id={{CHECKOUT_SESSION_ID}}",
        cancel_url=f"{BACKEND_URL}/payments/cancel",
        metadata={"user_id": str(user["id"]), "credits": body.credits * 10},
    )
    return {"url": session.url}


@app.get("/payments/success")
def payment_success(session_id: str):
    return {"message": "Paiement reçu ! Tes crédits ont été ajoutés. Tu peux fermer cette page."}


@app.get("/payments/cancel")
def payment_cancel():
    return {"message": "Paiement annulé."}


@app.post("/payments/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")
    try:
        event = stripe.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)
    except Exception:
        raise HTTPException(status_code=400, detail="Webhook invalide")

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        if session["payment_status"] == "paid":
            user_id = session["metadata"]["user_id"]
            credits_to_add = int(session["metadata"]["credits"])
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE users SET credits = credits + %s WHERE id = %s",
                        (credits_to_add, user_id),
                    )
                    cur.execute(
                        "INSERT INTO payments (user_id, stripe_session_id, credits_purchased, amount_cents, status) "
                        "VALUES (%s, %s, %s, %s, 'paid')",
                        (user_id, session["id"], credits_to_add, session["amount_total"]),
                    )
                conn.commit()
    return {"ok": True}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
