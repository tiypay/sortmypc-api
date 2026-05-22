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
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
SUPABASE_DB_URL   = os.getenv("SUPABASE_DB_URL", "")
JWT_SECRET        = os.getenv("JWT_SECRET", "change-me")
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PRICE_ID      = os.getenv("STRIPE_PRICE_ID", "")
STRIPE_PRICE_ID_1    = os.getenv("STRIPE_PRICE_ID_1", "price_1TWUFvLm8MCEET5iwA1Pu1uR")
BACKEND_URL       = os.getenv("BACKEND_URL", "http://localhost:8000")
RESEND_API_KEY    = os.getenv("RESEND_API_KEY", "")
FROM_EMAIL        = os.getenv("FROM_EMAIL", "SortMyPC <noreply@raw-x.fr>")

AI_MODEL       = "claude-haiku-4-5-20251001"
FILES_PER_CREDIT = 100
JWT_EXPIRE_DAYS  = 30
MAX_IMAGE_PX     = 150
MAX_IMAGES_CALL  = 100
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
                    credits INTEGER DEFAULT 3,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                ALTER TABLE users ALTER COLUMN credits SET DEFAULT 3;
                CREATE TABLE IF NOT EXISTS payments (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID REFERENCES users(id),
                    stripe_session_id TEXT UNIQUE,
                    credits_purchased INTEGER,
                    amount_cents INTEGER,
                    status TEXT DEFAULT 'pending',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE TABLE IF NOT EXISTS referrals (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    email TEXT UNIQUE NOT NULL,
                    ref_code TEXT UNIQUE NOT NULL,
                    referred_by TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE TABLE IF NOT EXISTS promo_codes (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    code TEXT UNIQUE NOT NULL,
                    email TEXT NOT NULL,
                    credits INTEGER NOT NULL,
                    rank TEXT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    redeemed_at TIMESTAMPTZ,
                    redeemed_by TEXT
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
    folder: str | None = None   # dossier parent pour contexte

class SortRequest(BaseModel):
    files: list[FileItem]

class CheckoutRequest(BaseModel):
    credits: int   # number of credit packs to buy
    pack: str = "10"  # "10" = 10 crédits à 2.99€, "1" = 1 crédit à 1€

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
    "Tu es un expert en organisation de fichiers et dossiers. "
    "Tu reçois une liste d'éléments (fichiers et dossiers). "
    "Propose une structure HIÉRARCHIQUE avec 10 à 20 grandes catégories au niveau 1. "
    "Chaque catégorie de niveau 1 peut contenir des sous-catégories (niveau 2) si utile. "
    "RÈGLES DE NOMMAGE OBLIGATOIRES : "
    "utilise des espaces (jamais d'underscores ni de tirets), première lettre majuscule, noms courts en français. "
    "Réponds UNIQUEMENT en JSON valide, sans texte avant ni après. "
    'Format : {"Categorie1": {"Sous Cat1": ["f1.pdf", "f2.jpg"], "Sous Cat2": ["f3.mp4"]}, '
    '"Categorie2": ["f4.docx", "f5.txt"]}. '
    "Les valeurs de niveau 1 sont soit un objet de sous-catégories, soit une liste directe. "
    "Noms courts et clairs en français. Mets les inclassables dans 'Divers'."
)


def _parse_json(raw: str) -> dict:
    import re

    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    def _try(s: str):
        try:
            return json.loads(s)
        except json.JSONDecodeError as e:
            if "Extra data" in str(e):
                try:
                    return json.loads(s[:e.pos].strip())
                except Exception:
                    pass
        return None

    # 1. Tentative directe
    r = _try(raw)
    if r is not None:
        return r

    # 2. Fix backslashes invalides (chemins Windows)
    fixed = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw)
    r = _try(fixed)
    if r is not None:
        return r

    # 3. Fix virgules manquantes entre éléments JSON (bug fréquent de Claude)
    #    "valeur1"\n"valeur2"  →  "valeur1",\n"valeur2"
    fixed = re.sub(r'"\s*\n(\s*)"', r'",\n\1"', fixed)
    r = _try(fixed)
    if r is not None:
        return r

    #    ]\n"clé"  ou  }\n"clé"  →  ],\n"clé"
    fixed = re.sub(r'([\]\}])\s*\n(\s*)"', r'\1,\n\2"', fixed)
    r = _try(fixed)
    if r is not None:
        return r

    # 4. Supprimer les virgules en trop (trailing commas)
    fixed = re.sub(r',\s*([\]\}])', r'\1', fixed)
    r = _try(fixed)
    if r is not None:
        return r

    # 5. Dernier recours : extraire le premier bloc {...} valide
    match = re.search(r'\{.*\}', fixed, re.DOTALL)
    if match:
        r = _try(match.group())
        if r is not None:
            return r

    raise ValueError(f"JSON invalide après toutes les corrections : {raw[:200]}")


def _sort_batch(client: anthropic.Anthropic, batch: list[FileItem]) -> dict:
    intro_lines = ["Fichiers à organiser :\n"]
    image_blocks = []
    images_used = 0

    for f in batch:
        ctx = f" (📂 {f.folder})" if f.folder else ""
        if f.image_b64 and images_used < MAX_IMAGES_CALL:
            image_blocks.append({"type": "text", "text": f"\n[{f.name}]{ctx} :"})
            image_blocks.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": f.image_b64},
            })
            images_used += 1
        elif f.text_preview:
            intro_lines.append(f"[{f.name}]{ctx} → {f.text_preview[:150]}")
        else:
            intro_lines.append(f"{f.name}{ctx}")

    content = [{"type": "text", "text": "\n".join(intro_lines)}] + image_blocks

    msg = client.messages.create(
        model=AI_MODEL, max_tokens=8192, system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )
    return _parse_json(msg.content[0].text)


def _normalize(s: str) -> str:
    """Normalize a category name for deduplication: lowercase, spaces instead of _/-."""
    return ' '.join(s.replace('_', ' ').replace('-', ' ').split()).lower()


def _deep_merge(base: dict, patch: dict) -> dict:
    """Merge patch into base with normalized key matching to prevent near-duplicate categories."""
    norm_map = {_normalize(k): k for k in base}

    for k, v in patch.items():
        existing = norm_map.get(_normalize(k))

        if existing is None:
            base[k] = v
            norm_map[_normalize(k)] = k
        elif isinstance(base[existing], dict) and isinstance(v, dict):
            _deep_merge(base[existing], v)
        elif isinstance(base[existing], list) and isinstance(v, list):
            base[existing].extend(v)
        elif isinstance(base[existing], list) and isinstance(v, dict):
            old = base[existing]
            base[existing] = dict(v)
            if old:
                base[existing].setdefault("Divers", []).extend(old)
        elif isinstance(base[existing], dict) and isinstance(v, list):
            base[existing].setdefault("Divers", []).extend(v)

    return base


CONSOLIDATION_PROMPT = (
    "Tu reçois une liste de noms de catégories de fichiers générés par plusieurs analyses indépendantes. "
    "Ta mission : regrouper en 10 à 15 grandes catégories unifiées. "
    "RÈGLES STRICTES : "
    "1. Fusionne TOUTES les catégories similaires ou qui se chevauchent, même si les noms diffèrent légèrement "
    "(ex: 'Cours_Maths', 'Cours - Mathématiques', 'Mathématiques Cours', 'Cours Mathématiques' → 'Mathématiques'). "
    "2. Fusionne les catégories dont les thèmes se recoupent "
    "(ex: 'Ressources Multimédia' + 'Multimédia' → 'Multimédia'). "
    "3. N'utilise 'Divers' QUE pour les fichiers vraiment inclassables — si une catégorie a un thème clair, "
    "elle doit aller dans la bonne grande catégorie, jamais dans Divers. "
    "4. Noms avec espaces (jamais d'underscores), première lettre majuscule, courts, en français. "
    "Réponds UNIQUEMENT en JSON valide, sans texte avant ni après. "
    'Format : {"GrandeCategorie": ["ancienneCat1", "ancienneCat2"], ...}. '
    "Chaque ancienne catégorie doit apparaître exactement une fois."
)


def _consolidate(client: anthropic.Anthropic, merged: dict) -> dict:
    """Second AI pass: collapse 50-200 redundant categories into 10-20 unified ones."""
    if len(merged) <= 20:
        return merged  # Already clean, skip

    category_names = list(merged.keys())
    msg = client.messages.create(
        model=AI_MODEL,
        max_tokens=8192,
        system=CONSOLIDATION_PROMPT,
        messages=[{"role": "user", "content": json.dumps(category_names, ensure_ascii=False)}],
    )
    mapping = _parse_json(msg.content[0].text)

    new_merged: dict = {}
    mapped_old: set = set()

    for new_cat, old_cats in mapping.items():
        new_merged[new_cat] = {}
        for old_cat in old_cats:
            if old_cat not in merged:
                continue
            mapped_old.add(old_cat)
            val = merged[old_cat]
            if isinstance(val, list):
                new_merged[new_cat].setdefault("Divers", []).extend(val)
            elif isinstance(val, dict):
                _deep_merge(new_merged[new_cat], val)
        # If subcategory ended up with only a "Divers" key and nothing else, flatten it
        if list(new_merged[new_cat].keys()) == ["Divers"]:
            new_merged[new_cat] = new_merged[new_cat]["Divers"]

    # Unmapped categories go to Divers
    for old_cat, val in merged.items():
        if old_cat not in mapped_old:
            divers = new_merged.setdefault("Divers", {})
            if isinstance(val, list):
                divers.setdefault("Divers", []).extend(val)
            elif isinstance(val, dict):
                _deep_merge(divers, val)

    return new_merged


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
        # Trier par dossier parent pour que les fichiers liés arrivent dans le même batch
        files_ordered = sorted(body.files, key=lambda f: (f.folder or '').lower())
        batches = [files_ordered[i:i + BATCH_SIZE] for i in range(0, len(files_ordered), BATCH_SIZE)]
        merged: dict = {}
        for batch in batches:
            plan = _sort_batch(ai_client, batch)
            _deep_merge(merged, plan)
        # Second pass: consolidate redundant categories into 10-20 unified ones
        merged = _consolidate(ai_client, merged)
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
    try:
        if body.pack == "1":
            price_id = STRIPE_PRICE_ID_1
            credits_to_add = body.credits * 1
        else:
            price_id = STRIPE_PRICE_ID
            credits_to_add = body.credits * 10
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{"price": price_id, "quantity": body.credits}],
            mode="payment",
            success_url=f"{BACKEND_URL}/payments/success?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{BACKEND_URL}/payments/cancel",
            metadata={"user_id": str(user["id"]), "credits": credits_to_add},
        )
        return {"url": session.url}
    except stripe.error.StripeError as e:
        raise HTTPException(status_code=500, detail=f"Erreur Stripe : {e.user_message or str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur paiement : {str(e)}")


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


# ── Referral system ───────────────────────────────────────────────────────────

import secrets
import string

RANK_THRESHOLDS = [
    (100, "legend",     720),  # 100 invités → 1 mois illimité (~720 crédits)
    (50,  "diamond",    20),   # 50  invités → 20 crédits bonus
    (25,  "blazer",     8),    # 25  invités → 8  crédits bonus
    (10, "starlighter",  3),   # 10 invités → 3  crédits bonus
]

def _gen_ref_code(n=8):
    chars = string.ascii_uppercase + string.digits
    return ''.join(secrets.choice(chars) for _ in range(n))


RANK_PREFIXES = {
    "starlighter": "STAR",
    "blazer":      "BLAZ",
    "diamond":     "DIAM",
    "legend":      "LGND",
}

RANK_LABELS = {
    "starlighter": "Starlighter ⭐",
    "blazer":      "Blazer 🔥",
    "diamond":     "Diamond 💎",
    "legend":      "Legend 👑",
}


def _gen_promo_code(rank: str) -> str:
    prefix = RANK_PREFIXES.get(rank, "PRMO")
    suffix = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    return f"{prefix}-{suffix}"


def _send_promo_email(to_email: str, rank: str, credits: int, code: str):
    """Send promo code email via Resend API."""
    if not RESEND_API_KEY:
        print(f"[EMAIL SKIP] No RESEND_API_KEY — code for {to_email}: {code}")
        return
    try:
        import urllib.request
        rank_label = RANK_LABELS.get(rank, rank.capitalize())
        subject = f"🎉 Tu as atteint {rank_label} — voici ton code promo SortMyPC"
        html = f"""
<!DOCTYPE html>
<html lang="fr">
<head><meta charset="UTF-8"></head>
<body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
             background:#0D1117;color:#E6EDF3;margin:0;padding:40px 0;">
  <div style="max-width:480px;margin:0 auto;background:#161B22;border-radius:16px;
              border:1px solid #21262D;overflow:hidden;">
    <div style="background:linear-gradient(135deg,#2979FF,#00C853);padding:24px 32px;">
      <h1 style="margin:0;font-size:22px;color:#fff">🎉 Nouveau rang débloqué !</h1>
      <p style="margin:6px 0 0;color:rgba(255,255,255,.85);font-size:14px">
        Tu as atteint <strong>{rank_label}</strong> sur SortMyPC
      </p>
    </div>
    <div style="padding:32px;">
      <p style="margin:0 0 20px;color:#8B949E;font-size:14px">
        Félicitations ! Tu as parrainé suffisamment d'amis pour monter de rang.<br>
        Voici ton récompense : <strong style="color:#E6EDF3">+{credits} crédit(s)</strong> offerts.
      </p>
      <p style="margin:0 0 8px;font-size:13px;color:#8B949E;
                letter-spacing:.06em;text-transform:uppercase;font-weight:700">
        Ton code promo
      </p>
      <div style="background:#0D1117;border:1.5px solid #2979FF;border-radius:10px;
                  padding:16px 24px;text-align:center;margin-bottom:24px;">
        <span style="font-size:26px;font-weight:900;letter-spacing:.12em;
                     color:#fff;font-family:monospace">{code}</span>
      </div>
      <p style="margin:0 0 20px;font-size:13px;color:#8B949E;">
        Ouvre l'application <strong style="color:#E6EDF3">SortMyPC</strong>,
        clique sur <strong style="color:#E6EDF3">🎁 Code promo</strong> dans le header
        et saisis ce code pour recevoir tes crédits.
      </p>
      <a href="https://apps.microsoft.com/store/detail/sortmypc/9MZVN3MF00LS"
         style="display:block;background:#2979FF;color:#fff;text-decoration:none;
                border-radius:10px;padding:12px;text-align:center;
                font-weight:700;font-size:14px;">
        Ouvrir SortMyPC
      </a>
    </div>
    <div style="padding:16px 32px;border-top:1px solid #21262D;
                font-size:12px;color:#4B5563;text-align:center;">
      SortMyPC · raw-x.fr · Ce code est à usage unique.
    </div>
  </div>
</body>
</html>"""
        payload = json.dumps({
            "from": FROM_EMAIL,
            "to": [to_email],
            "subject": subject,
            "html": html,
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.resend.com/emails",
            data=payload,
            headers={
                "Authorization": f"Bearer {RESEND_API_KEY}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(f"[EMAIL OK] Sent to {to_email} (rank={rank}) → {resp.status}")
    except Exception as e:
        print(f"[EMAIL ERR] {e} — code was: {code}")


class ReferralRegister(BaseModel):
    email: EmailStr
    referred_by: str | None = None


@app.post("/referral/register", status_code=201)
def referral_register(body: ReferralRegister, bg: BackgroundTasks):
    """Register an email for the referral program. Returns a unique ref_code."""
    promo_task = None  # (email, rank, credits, promo_code) to send after commit

    with get_conn() as conn:
        with conn.cursor() as cur:
            # Check if already registered
            cur.execute("SELECT ref_code FROM referrals WHERE email = %s", (body.email,))
            existing = cur.fetchone()
            if existing:
                raise HTTPException(status_code=409, detail="Email déjà inscrit")

            # Generate unique ref code
            for _ in range(10):
                code = _gen_ref_code()
                cur.execute("SELECT 1 FROM referrals WHERE ref_code = %s", (code,))
                if not cur.fetchone():
                    break

            # Insert
            cur.execute(
                "INSERT INTO referrals (email, ref_code, referred_by) VALUES (%s, %s, %s)",
                (body.email, code, body.referred_by)
            )

            # Check if referrer hits a new rank threshold → generate promo code
            if body.referred_by:
                cur.execute(
                    "SELECT email FROM referrals WHERE ref_code = %s",
                    (body.referred_by,)
                )
                referrer = cur.fetchone()
                if referrer:
                    referrer_email = referrer["email"]
                    cur.execute(
                        "SELECT COUNT(*) as cnt FROM referrals WHERE referred_by = %s",
                        (body.referred_by,)
                    )
                    new_count = (cur.fetchone()["cnt"] or 0) + 1

                    for (threshold, rank, bonus) in RANK_THRESHOLDS:
                        if new_count == threshold:
                            # Check we haven't already sent a promo for this rank
                            cur.execute(
                                "SELECT 1 FROM promo_codes WHERE email = %s AND rank = %s",
                                (referrer_email, rank)
                            )
                            if not cur.fetchone():
                                promo_code = _gen_promo_code(rank)
                                cur.execute(
                                    """INSERT INTO promo_codes (code, email, credits, rank)
                                       VALUES (%s, %s, %s, %s)""",
                                    (promo_code, referrer_email, bonus, rank)
                                )
                                promo_task = (referrer_email, rank, bonus, promo_code)
                            break

        conn.commit()

    # Send email outside the transaction
    if promo_task:
        bg.add_task(_send_promo_email, *promo_task)

    return {"ref_code": code, "email": body.email}


class PromoRedeem(BaseModel):
    code: str


@app.post("/promo/redeem")
def promo_redeem(body: PromoRedeem, user: dict = Depends(get_current_user)):
    """Validate and redeem a promo code. Adds credits to the authenticated user."""
    code = body.code.strip().upper()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM promo_codes WHERE code = %s", (code,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Code invalide ou inexistant.")
            if row["redeemed_at"]:
                raise HTTPException(status_code=409, detail="Ce code a déjà été utilisé.")
            credits = row["credits"]
            # Mark as redeemed
            cur.execute(
                "UPDATE promo_codes SET redeemed_at = NOW(), redeemed_by = %s WHERE code = %s",
                (user["email"], code)
            )
            # Add credits to the user
            cur.execute(
                "UPDATE users SET credits = credits + %s WHERE id = %s RETURNING credits",
                (credits, user["id"])
            )
            new_balance = cur.fetchone()["credits"]
        conn.commit()
    return {
        "credits": credits,
        "credits_remaining": new_balance,
        "rank": row["rank"],
        "code": code,
    }


@app.get("/referral/stats/{code_or_email}")
def referral_stats(code_or_email: str):
    """Get referral stats by ref_code or email."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            # Try by ref_code first, then email
            cur.execute(
                "SELECT * FROM referrals WHERE ref_code = %s OR email = %s",
                (code_or_email, code_or_email)
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Introuvable")

            cur.execute(
                "SELECT COUNT(*) as cnt FROM referrals WHERE referred_by = %s",
                (row["ref_code"],)
            )
            count = cur.fetchone()["cnt"] or 0

    return {
        "ref_code": row["ref_code"],
        "email": row["email"],
        "referral_count": count,
        "created_at": str(row["created_at"]),
    }


@app.get("/download/SortMyPC.exe")
async def download_exe():
    """Proxy the SortMyPC.exe installer without any client-side redirect."""
    import httpx
    DIRECT_URL = "https://github.com/tiypay/sortmypc-desktop/releases/download/v1.4.0/SortMyPC.exe"

    async def stream():
        async with httpx.AsyncClient(follow_redirects=True, timeout=120) as client:
            async with client.stream("GET", DIRECT_URL) as resp:
                async for chunk in resp.aiter_bytes(65536):
                    yield chunk

    return StreamingResponse(
        stream(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=SortMyPC.exe"},
    )


@app.get("/privacy", response_class=HTMLResponse)
def privacy_policy():
    return """<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Politique de confidentialité — SortMyPC</title>
  <style>
    body { font-family: Segoe UI, Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; color: #222; line-height: 1.7; }
    h1 { color: #1a1a2e; } h2 { color: #16213e; margin-top: 30px; }
    a { color: #3b82f6; } footer { margin-top: 40px; color: #888; font-size: 0.9em; }
  </style>
</head>
<body>
  <h1>Politique de confidentialité — SortMyPC</h1>
  <p><em>Dernière mise à jour : mai 2026</em></p>

  <p>SortMyPC est une application Windows qui utilise l'intelligence artificielle pour organiser vos fichiers.
  Cette politique explique comment nous collectons et utilisons vos données.</p>

  <h2>1. Données collectées</h2>
  <ul>
    <li><strong>Adresse e-mail</strong> : lors de la création de votre compte.</li>
    <li><strong>Métadonnées de fichiers</strong> : noms, extensions et aperçus de vos fichiers (jamais le contenu complet). Ces données sont envoyées à l'API Claude (Anthropic) pour générer le plan de tri.</li>
    <li><strong>Données de paiement</strong> : traitées exclusivement par Stripe. Nous ne stockons aucune information bancaire.</li>
    <li><strong>Historique d'utilisation</strong> : nombre de crédits utilisés.</li>
  </ul>

  <h2>2. Utilisation des données</h2>
  <p>Vos données sont utilisées uniquement pour faire fonctionner l'application : tri de fichiers par IA, gestion de votre compte et traitement des paiements. Nous ne vendons ni ne partageons vos données avec des tiers à des fins publicitaires.</p>

  <h2>3. Services tiers</h2>
  <ul>
    <li><strong>Anthropic (Claude AI)</strong> : traitement des métadonnées pour le tri. <a href="https://www.anthropic.com/privacy">Politique Anthropic</a></li>
    <li><strong>Stripe</strong> : paiements sécurisés. <a href="https://stripe.com/fr/privacy">Politique Stripe</a></li>
    <li><strong>Supabase</strong> : base de données hébergée. <a href="https://supabase.com/privacy">Politique Supabase</a></li>
  </ul>

  <h2>4. Conservation des données</h2>
  <p>Vos données sont conservées tant que votre compte est actif. Vous pouvez demander la suppression de votre compte en nous contactant.</p>

  <h2>5. Vos droits</h2>
  <p>Conformément au RGPD, vous disposez d'un droit d'accès, de rectification et de suppression de vos données. Contactez-nous à : <a href="mailto:support@sortmypc.app">support@sortmypc.app</a></p>

  <h2>6. Contact</h2>
  <p>Pour toute question : <a href="mailto:support@sortmypc.app">support@sortmypc.app</a></p>

  <footer>© 2026 SortMyPC. Tous droits réservés.</footer>
</body>
</html>"""


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
