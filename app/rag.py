import json
import math
import os
import re
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from groq import Groq

from app.database import get_connection

load_dotenv()

VECTOR_SIZE = 384


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _hash_token(token: str) -> int:
    h = 2166136261
    for ch in token:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def generate_embedding(text: str) -> list[float]:
    normalized = normalize_text(text)
    vector = [0.0] * VECTOR_SIZE
    if not normalized:
        return vector

    words = normalized.split()
    for word in words:
        vector[_hash_token(f"w:{word}") % VECTOR_SIZE] += 2.0
        padded = f" {word} "
        for n in (3, 4, 5):
            if len(padded) >= n:
                for i in range(len(padded) - n + 1):
                    ngram = padded[i : i + n]
                    vector[_hash_token(f"ng:{ngram}") % VECTOR_SIZE] += 0.5

    for i in range(len(words) - 1):
        bigram = f"{words[i]}_{words[i + 1]}"
        vector[_hash_token(f"bg:{bigram}") % VECTOR_SIZE] += 1.5

    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


STOPWORDS_ID = {
    "apa", "itu", "ini", "yang", "dan", "atau", "di", "ke", "dari", "untuk",
    "dengan", "adalah", "ialah", "merupakan", "ya", "kah", "dong", "sih", "nih",
    "bagaimana", "gimana", "kenapa", "mengapa", "bedanya", "perbedaan", "sama",
    "antara", "cara", "berapa", "kapan", "siapa", "dimana", "mana", "saja",
    "sebutkan", "jelaskan", "tolong", "coba", "bisa", "mau", "tanya", "tentang",
    "soal", "mengenai", "terkait", "kalau", "kalo", "jika", "bila", "apakah",
}


def lexical_bonus(question: str, title: str, chunk_text: str) -> float:
    raw_words = normalize_text(question).split()
    q_words = [w for w in raw_words if len(w) >= 2 and w not in STOPWORDS_ID]
    if not q_words:
        q_words = [w for w in raw_words if len(w) >= 2]
    if not q_words:
        return 0.0

    norm_title = normalize_text(title)
    norm_chunk = normalize_text(chunk_text)
    title_tokens = set(norm_title.split())
    chunk_tokens = set(norm_chunk.split())

    title_hits = sum(1 for w in q_words if w in title_tokens)
    text_hits = sum(1 for w in q_words if w in chunk_tokens)

    phrase = " ".join(q_words)
    phrase_bonus = 0.25 if len(q_words) >= 2 and phrase in norm_chunk else 0.0

    return (
        min(text_hits / len(q_words), 1.0) * 0.25
        + min(title_hits / len(q_words), 1.0) * 0.20
        + phrase_bonus
    )


def _split_long_block(block: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    if len(block) <= chunk_size:
        return [block]

    chunks: list[str] = []
    start = 0
    length = len(block)

    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            min_split = start + int(chunk_size * 0.5)
            split_pos = block.rfind("\n\n", min_split, end)
            if split_pos == -1:
                split_pos = block.rfind("\n", min_split, end)
            if split_pos == -1:
                split_pos = block.rfind(" ", min_split, end)

            if split_pos != -1:
                end = split_pos
            else:
                while end < length and not block[end].isspace():
                    end += 1

        piece = block[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= length:
            break

        next_start = max(start + 1, end - overlap)
        if next_start > 0 and not block[next_start - 1].isspace():
            while next_start < length and not block[next_start].isspace():
                next_start += 1
            while next_start < length and block[next_start].isspace():
                next_start += 1

        start = next_start if next_start < length else end

    return chunks


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    clean = re.sub(r"\r\n?", "\n", text).strip()
    clean = re.sub(r"\n{3,}", "\n\n", clean)
    if not clean:
        return []

    # Kelompokkan teks menjadi blok-blok logis agar tabel Markdown tidak pernah terpotong
    blocks: list[tuple[str, bool]] = []  # (content, is_table)
    lines = clean.split("\n")
    buf: list[str] = []
    in_table = False

    def flush_buf(is_tbl: bool) -> None:
        if buf:
            content = "\n".join(buf).strip()
            if content:
                blocks.append((content, is_tbl))
            buf.clear()

    for line in lines:
        stripped = line.strip()
        is_tbl_line = stripped.startswith("|") and stripped.endswith("|") and len(stripped) > 2
        if is_tbl_line:
            if not in_table:
                flush_buf(False)
                in_table = True
            buf.append(line)
        else:
            if in_table:
                flush_buf(True)
                in_table = False
            if stripped == "":
                flush_buf(False)
            else:
                buf.append(line)
    flush_buf(in_table)

    # Gabungkan blok-blok ke dalam chunk hingga chunk_size; blok tabel dibiarkan utuh
    chunks: list[str] = []
    current = ""

    for content, is_tbl in blocks:
        if is_tbl:
            if current and len(current) + 2 + len(content) <= max(chunk_size, 1400):
                current = f"{current}\n\n{content}"
            else:
                if current:
                    chunks.append(current)
                    current = ""
                if len(content) <= 2000:
                    chunks.append(content)
                else:
                    # Jika tabel sangat besar, pisah per baris sambil mempertahankan header tabel
                    tbl_lines = content.split("\n")
                    header = "\n".join(tbl_lines[:2]) if len(tbl_lines) >= 2 else ""
                    sub = header
                    for row in tbl_lines[2:]:
                        if len(sub) + 1 + len(row) > chunk_size and sub != header:
                            chunks.append(sub)
                            sub = f"{header}\n{row}" if header else row
                        else:
                            sub = f"{sub}\n{row}" if sub else row
                    if sub:
                        chunks.append(sub)
            continue

        if len(content) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_long_block(content, chunk_size=chunk_size, overlap=overlap))
            continue

        if not current:
            current = content
        elif len(current) + 2 + len(content) <= chunk_size:
            current = f"{current}\n\n{content}"
        else:
            chunks.append(current)
            current = content

    if current:
        chunks.append(current)

    return chunks


def search_chunks_by_role(
    question: str,
    role_name: str,
    top_k: int = 4,
    min_score: float = 0.16,
) -> list[dict[str, Any]]:
    if not question.strip():
        return []

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            dc.id AS chunk_id,
            dc.document_id,
            dc.chunk_index,
            dc.text,
            dc.embedding_json,
            d.title AS document_title,
            d.filename AS document_filename
        FROM document_chunks dc
        JOIN documents d ON d.id = dc.document_id
        JOIN document_roles dr ON dr.document_id = d.id
        JOIN roles r ON r.id = dr.role_id
        WHERE LOWER(r.name) = LOWER(?) AND d.is_active = 1
        """,
        (role_name.strip(),),
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return []

    q_embedding = generate_embedding(question)
    scored: list[dict[str, Any]] = []

    for row in rows:
        emb = json.loads(row["embedding_json"])
        vec_score = cosine_similarity(q_embedding, emb)
        bonus = lexical_bonus(question, row["document_title"], row["text"])
        total_score = vec_score + bonus

        if total_score >= min_score:
            scored.append(
                {
                    "chunk_id": row["chunk_id"],
                    "document_id": row["document_id"],
                    "document_title": row["document_title"],
                    "document_filename": row["document_filename"],
                    "chunk_index": row["chunk_index"],
                    "text": row["text"],
                    "score": round(total_score, 4),
                }
            )

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def get_unique_sources(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[int, dict[str, Any]] = {}
    for ch in chunks:
        doc_id = ch["document_id"]
        if doc_id not in seen:
            seen[doc_id] = {
                "document_id": doc_id,
                "title": ch["document_title"],
                "filename": ch["document_filename"],
                "url": f"/api/public/documents/{doc_id}/file",
                "score": ch["score"],
            }
    return list(seen.values())


GREETING_PATTERNS = {
    "hai", "halo", "hallo", "hi", "hello", "hey", "p",
    "pagi", "selamat pagi", "siang", "selamat siang",
    "sore", "selamat sore", "malam", "selamat malam",
    "assalamualaikum", "assalamu'alaikum", "permisi",
    "apa kabar", "siapa kamu", "kamu siapa",
    "terima kasih", "makasih", "thanks", "thank you", "oke", "ok", "sip",
}


def is_greeting_or_smalltalk(question: str) -> bool:
    cleaned = normalize_text(question)
    if not cleaned:
        return False
    if cleaned in GREETING_PATTERNS:
        return True
    words = cleaned.split()
    if len(words) <= 3 and any(w in {"hai", "halo", "hallo", "hi", "hello", "assalamualaikum", "makasih", "thanks"} for w in words):
        return True
    return False


def _safe_int(val: Any) -> int | None:
    try:
        return int(val) if val is not None and str(val).strip() != "" else None
    except (ValueError, TypeError):
        return None


def _log_groq_usage(
    model: str,
    completion: Any | None = None,
    headers: Any | None = None,
    status: str = "success",
) -> None:
    try:
        usage = getattr(completion, "usage", None) if completion else None
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)

        h = {k.lower(): v for k, v in headers.items()} if headers else {}
        limit_requests = _safe_int(h.get("x-ratelimit-limit-requests"))
        remaining_requests = _safe_int(h.get("x-ratelimit-remaining-requests"))
        reset_requests = str(h.get("x-ratelimit-reset-requests", "") or "")
        limit_tokens = _safe_int(h.get("x-ratelimit-limit-tokens"))
        remaining_tokens = _safe_int(h.get("x-ratelimit-remaining-tokens"))
        reset_tokens = str(h.get("x-ratelimit-reset-tokens", "") or "")

        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO groq_usage_logs (
                model, prompt_tokens, completion_tokens, total_tokens,
                limit_requests, remaining_requests, reset_requests,
                limit_tokens, remaining_tokens, reset_tokens,
                status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                limit_requests,
                remaining_requests,
                reset_requests,
                limit_tokens,
                remaining_tokens,
                reset_tokens,
                status,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def generate_answer(question: str, role_name: str, relevant_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    if not relevant_chunks:
        if is_greeting_or_smalltalk(question):
            cleaned = normalize_text(question)
            if any(w in cleaned for w in ("terima kasih", "makasih", "thanks", "thank")):
                return {
                    "answer": "Sama-sama! Jika ada pertanyaan lain seputar layanan atau dokumen akademik, silakan tanyakan ya.",
                    "sources": [],
                }
            return {
                "answer": "Halo! Saya Asisten Dokumen Akademik. Silakan tanyakan informasi seputar layanan akademik, jam operasional, alur menghubungi SSC, atau link penting lainnya.",
                "sources": [],
            }
        return {
            "answer": (
                "Maaf, informasi yang Anda tanyakan belum tersedia saat ini."
            ),
            "sources": [],
        }

    sources = get_unique_sources(relevant_chunks)
    context = "\n\n---\n\n".join(
        f"[{c['document_title']}]\n{c['text']}" for c in relevant_chunks
    )

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or api_key == "your_groq_api_key_here":
        return {
            "answer": (
                "GROQ_API_KEY belum dikonfigurasi di file .env, tetapi sistem berhasil menemukan "
                f"{len(relevant_chunks)} potongan informasi yang relevan untuk role '{role_name}'."
            ),
            "sources": sources,
        }

    model = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    client = Groq(api_key=api_key)

    system_prompt = (
        "Kamu adalah Asisten AI Layanan Akademik (SSC) yang ramah, konsisten, dan akurat. "
        "Jawablah pertanyaan secara langsung hanya berdasarkan informasi referensi yang disediakan, tanpa mengarang fakta baru. "
        "ATURAN KONSISTENSI: "
        "1) Jika terdapat beberapa versi aturan (misalnya tahun 2022 dan 2025), utamakan ketentuan terbaru (2025) sebagai acuan utama, kecuali pengguna secara eksplisit meminta perbandingan antar versi. "
        "2) Jika istilah yang ditanyakan pengguna merujuk pada konsep yang jelas dijelaskan dalam referensi (misalnya 'TAK Wajib' merujuk pada kategori kegiatan wajib TAK), jelaskan langsung definisi dan rinciannya dengan tegas dan konsisten—jangan menyatakan istilah tersebut 'tidak didefinisikan' jika penjelasannya ada di referensi. "
        "3) PENTING: Jangan pernah menyebut frasa seperti 'berdasarkan dokumen', 'dalam dokumen', 'menurut konteks', 'dokumen yang diberikan', atau merujuk pada kata 'dokumen'/'konteks' dalam jawabanmu. "
        "Sampaikan informasi seolah-olah itu adalah pengetahuan langsung layanan SSC. "
        "Jika informasi yang ditanyakan benar-benar tidak tersedia pada referensi, katakan dengan sopan dan singkat bahwa informasi tersebut belum tersedia saat ini. "
        "Jangan gunakan sintaks LaTeX seperti $...$, $$...$$, \\frac, atau \\times; "
        "tuliskan rumus matematika dalam teks biasa yang bersih (contoh: IKK = (Ni / Nm) × 4)."
    )

    user_prompt = (
        f"Pertanyaan: {question}\n\n"
        f"Referensi informasi:\n{context}\n\n"
        "Jawablah pertanyaan di atas secara langsung, ringkas, dan natural dalam bahasa Indonesia."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    used_model = model
    try:
        try:
            raw_resp = client.chat.completions.with_raw_response.create(
                model=model,
                temperature=0.0,
                max_tokens=700,
                messages=messages,
            )
            completion = raw_resp.parse()
            _log_groq_usage(used_model, completion=completion, headers=raw_resp.headers, status="success")
        except Exception as first_exc:
            if "model_not_found" not in str(first_exc):
                raise first_exc
            available_ids = [
                m.id for m in client.models.list().data
                if not any(skip in m.id.lower() for skip in ("whisper", "guard", "tts", "safeguard"))
            ]
            qwen_models = [mid for mid in available_ids if "qwen" in mid.lower()]
            fallback_model = qwen_models[0] if qwen_models else (available_ids[0] if available_ids else None)
            if not fallback_model:
                raise first_exc
            used_model = fallback_model
            raw_resp = client.chat.completions.with_raw_response.create(
                model=fallback_model,
                temperature=0.0,
                max_tokens=700,
                messages=messages,
            )
            completion = raw_resp.parse()
            _log_groq_usage(used_model, completion=completion, headers=raw_resp.headers, status="success")

        raw_answer = (completion.choices[0].message.content or "").strip()
        answer = re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL).strip()
        if not answer:
            answer = "Maaf, tidak ada respons yang dihasilkan oleh model AI."
        return {"answer": answer, "sources": sources}
    except Exception as exc:
        err_headers = getattr(getattr(exc, "response", None), "headers", None)
        _log_groq_usage(used_model, completion=None, headers=err_headers, status="error")
        return {
            "answer": f"Terjadi kendala saat menghubungi Groq API: {exc}",
            "sources": sources,
        }
