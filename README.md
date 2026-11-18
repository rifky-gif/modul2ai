# modul2ai — Chatbot RAG Berbasis Role (FastAPI + Groq)

## Menjalankan secara lokal
```bash
python -m venv venv
venv\Scripts\activate          # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
uvicorn app.main:app --reload
```
- Chatbot: http://127.0.0.1:8000  (pilih **Role** di pojok kanan atas)
- Admin: http://127.0.0.1:8000/admin  (login `admin` / `admin123`)

Saat pertama kali jalan, `dataset/` otomatis diindeks:
| Lokasi | Role yang boleh akses |
| --- | --- |
| `dataset/*` (SSC, SK TAK 2022, SK TAK 2025; data mahasiswa/umum) | `mahasiswa` dan `dosen` |
| `dataset/dosen/Panduan_Internal_Dosen.txt` | **hanya `dosen`** |

Dengan demikian, mahasiswa hanya melihat data mahasiswa/umum, sedangkan dosen dapat melihat data mahasiswa/umum dan data khusus dosen. Riwayat percakapan chat juga disimpan terpisah untuk setiap role.

Jika `modul2ai.db` sudah ada, seeding dilewati. Hapus file tersebut dan restart untuk mengulang.

## Pengujian Tugas 1 & 2 (RBAC)
1. Buka halaman chat, role **Mahasiswa**, tanyakan: *"Apa kode verifikasi internal dosen untuk pengujian praktikum?"* → tidak ada jawaban/sumber dari dokumen dosen.
2. Ganti role ke **Dosen**, tanyakan hal yang sama → jawaban memuat `DOSEN-RBAC-2026` dan sumber *Panduan Internal Dosen*.
3. Di `/admin`: tambah role baru, upload dokumen dengan memilih role, dan gunakan tombol **Ubah Role** pada dokumen.
4. Popup **Buka Chatbot** di admin punya pemilih role untuk menguji tiap role.

## Deployment ke Railway (Tugas 3)
0. Pastikan file `.python-version` (isi `3.12`) ikut ter-push.
1. Push project ke GitHub (`.env` sudah di-`.gitignore`, **jangan** di-commit).
2. Railway → New Project → Deploy from GitHub repo. `Procfile` sudah berisi start command.
3. Tab **Variables**, isi: `GROQ_API_KEY`, `GROQ_MODEL`, `JWT_SECRET` (acak, panjang), `ADMIN_USERNAME`, `ADMIN_PASSWORD` (**ganti dari admin123**).
4. Agar database & upload tidak hilang saat redeploy: tambahkan **Volume** (mount path `/data`) lalu set `DB_PATH=/data/modul2ai.db` dan `UPLOAD_DIR=/data/uploads`.
5. Settings → Networking → **Generate Domain**, lalu uji dan ambil tangkapan layar untuk laporan.

## Perubahan dari kode modul
- Role `dosen` ditambahkan; manajemen role (tambah/ubah/hapus) dan pemilihan role saat upload kini ada di panel admin; role dokumen bisa diubah setelah upload.
- **Perbaikan keamanan RBAC:** folder `/uploads` tidak lagi dibuka sebagai static publik (file khusus dosen sebelumnya bisa diunduh langsung lewat URL). File kini dilayani lewat `/api/public/documents/{id}/file?role=...` yang mengecek role dan status aktif dokumen.
- Validasi nama role, proteksi role bawaan `mahasiswa`, validasi role saat upload, dan dukungan `UPLOAD_DIR`.

## Catatan keamanan
Pemilihan role pada halaman chat bersifat sukarela (tanpa login), sesuai desain modul. Untuk produksi sungguhan, role `dosen` sebaiknya dilindungi login/kode akses.
