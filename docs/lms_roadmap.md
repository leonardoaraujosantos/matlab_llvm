# Learning Platform (LMS) — Roadmap

> A learning-management layer on top of the remote backend
> ([`docs/remote_backend_plan.md`](remote_backend_plan.md)): per-user data
> and files, **programming courses** with lessons and exercises, and
> per-user learning paths / scores. Code exercises are **auto-graded by the
> `matlabc` compiler itself** (run the submission, compare output).

**Status:** planned, not started. Captured 2026-05-22.

---

## 1. Locked decisions (from product review)

| Decision | Choice |
|---|---|
| Database | **PostgreSQL only** (no SQLite fallback) |
| ORM / migrations | **SQLAlchemy 2.0 async + asyncpg + Alembic**, driven by `DATABASE_URL` (mirrors the geo_dashboard backend) |
| Code grading | **Auto-grade via `matlabc`** — run the submission in the existing sandbox and compare stdout to expected output / test cases |
| Scope | **Full LMS incl. authoring** (learner flow + instructor/admin CRUD) |
| Identity | **CyberdyneAuth** stays the IdP; our DB stores app data keyed by the Cyberdyne user UUID, JIT-provisioned on first authenticated request. No passwords in our DB. |
| File storage | **Local per-user workspace** is the execution cwd (required); a **pluggable durable backend** sits behind it — `local` volume (default, single-node) or **MinIO/S3** via aioboto3 (`STORAGE_BACKEND=s3`, prod/scale). See §3.1. |

Local dev runs Postgres via a `docker-compose` service (no SQLite), since the
prod target is Postgres and we don't want engine drift.

---

## 2. Stack & layout

`DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/matlab_llvm` (Coolify
provisions the Postgres addon → injects the URL). Add under `server/`:

```
server/
  db/
    base.py          # DeclarativeBase + naming convention
    session.py       # async engine + sessionmaker + get_session dependency
    models.py        # ORM models (or split per aggregate)
  repositories/      # thin data-access helpers (or use the session directly)
  lms/
    grading.py       # auto-grader: run a submission via the sandbox, score it
    seed.py          # fixture loader for an intro course
  storage.py         # pluggable file store: local volume | s3/MinIO (aioboto3)
  routers/
    courses.py       # learner: browse / enroll
    exercises.py     # learner: fetch / submit (grade)
    progress.py      # learner: my learning path / submissions
    admin_courses.py # instructor/admin authoring CRUD
  alembic/           # migrations (env.py reads settings.database_url)
  alembic.ini
```

Config additions: `database_url`, `db_pool_size`, `db_echo`,
`grading_timeout_s`, `storage_backend` (`local|s3`), `s3_endpoint_url`,
`s3_bucket`, `s3_access_key`, `s3_secret_key`, `s3_region`. New dep for the
s3 path: `aioboto3` (+ `moto` for test mocking — matches geo_dashboard).
Health check (`/healthz`) gains a DB-connectivity probe.

---

## 3. Data model

CyberdyneAuth owns accounts; `users.id` **is** the Cyberdyne UUID.

```
users            id(uuid PK = cyberdyne id), email, display_name,
                 role(enum: student|instructor|admin, default student),
                 created_at, last_seen_at
user_files       id(uuid PK), user_id→users(cascade), rel_path, storage_key,
                 backend(local|s3), size_bytes, content_type, created_at,
                 updated_at, unique(user_id, rel_path)
                 # catalog; bytes live on the local volume and/or the object store
courses          id(uuid PK), slug(unique), title, description, language('matlab'),
                 ordinal, is_published(bool), created_by→users, created_at, updated_at
lessons          id(uuid PK), course_id→courses(cascade), title, body(markdown),
                 ordinal, created_at, updated_at
exercises        id(uuid PK), lesson_id→lessons(cascade),
                 kind(enum: code|mcq|short), prompt(markdown), starter_code,
                 expected_stdout, tests(jsonb), choices(jsonb), correct_answer,
                 grading(jsonb: trim/whitespace/tolerance), points(int), ordinal
submissions      id(uuid PK), user_id→users, exercise_id→exercises, payload,
                 is_correct, score, stdout, stderr, duration_ms, created_at
                 # index(user_id, exercise_id, created_at)
enrollments      id(uuid PK), user_id→users, course_id→courses,
                 status(enum: active|completed|dropped), enrolled_at, completed_at,
                 unique(user_id, course_id)
progress         user_id+exercise_id(composite PK), best_score, attempts,
                 completed(bool), completed_at, updated_at
                 # course-level progress derived by aggregation (or a view)
```

Roles default to `student` at provisioning; an admin can promote, and later we
can sync from CyberdyneAuth groups/policies (its API exposes
`/admin/groups`, `/users/{id}/iam`, `/admin/policies`).

### 3.1 Storage: per-user folders & MinIO/S3

Two layers, because **`matlabc` runs against a real POSIX directory** —
object storage cannot be a working directory:

- **Execution (local, required).** Every run uses a local per-user workspace
  `WORKSPACE_ROOT/<identity-uuid>/<session>/` — *already implemented*, keyed by
  the verified CyberdyneAuth principal, which overrides any client-supplied
  `user_id` so a user can never reach another's folder. (Verified live: the
  workspace dir is named by the identity UUID.) This stays even with S3.
- **Durability (object store, pluggable).** Persistent files (uploaded datasets,
  saved programs, kept artifacts) live under a per-user prefix
  `users/<identity-uuid>/…`. `STORAGE_BACKEND`:
  - `local` (default) — the mounted volume *is* the store. Fine for single-node
    / dev. The folder *is* the durable copy.
  - `s3` — MinIO / R2 / S3 via **aioboto3** (`S3_ENDPOINT_URL`, `S3_BUCKET`).
    The local workspace becomes a **scratch cache**: hydrate the needed objects
    in before a run, push new artifacts/saves back out after.

**Do we need MinIO?** Not for an MVP single-node deployment — the per-user
volume folder already works and is required as the execution cwd. Adopt
MinIO/S3 when you want any of: **horizontal scaling** (multiple backend replicas
can't share one local volume), **durability / backups / versioning**,
**presigned download URLs** (offload the API, CDN-friendly), or **large files /
many users**. A growing multi-user learning platform will want these, so the
storage layer is pluggable from day one — flip `STORAGE_BACKEND=s3` (with a
Coolify MinIO resource) and nothing else changes for callers.

`user_files.storage_key` + `backend` keep the catalog backend-agnostic;
isolation is always enforced by the auth principal, never the request body.

---

## 4. Auto-grading (the compiler is the grader)

`POST /v1/exercises/{id}/submit` with the user's payload:

1. Load the exercise.
2. **code** → run `payload` through the existing sandbox (`sandbox.run` /
   `services` — same rlimits + cwd jail + tier-2 sandbox + concurrency cap,
   with a `grading_timeout_s` wall clock). Capture stdout/stderr.
   - Normalize per `grading` opts (trim, collapse whitespace, numeric
     tolerance) and compare to `expected_stdout`; or, when `tests` is set,
     run each case (function-call assertions) — a richer runner (T4).
   - **mcq** → compare to `correct_answer`. **short** → normalized / regex match.
3. Persist a `submissions` row; upsert `progress` (`best_score = max`,
   `attempts += 1`, `completed` if correct); roll up `enrollments`.
4. Return `{is_correct, score, stdout, stderr, feedback}` — never leak
   `expected_stdout`/solutions to the learner.

Grading reuses the entire isolation stack — a submission is just a bounded,
sandboxed compiler run. The same engine that powers `/v1/repl` grades exercises.

---

## 5. API surface

**Learner** (any authenticated role)
```
GET  /v1/courses                  list published courses (+ my enrollment/progress)
GET  /v1/courses/{slug}           course detail: lessons + exercise summaries
POST /v1/courses/{slug}/enroll
GET  /v1/lessons/{id}             lesson body + its exercises
GET  /v1/exercises/{id}           prompt/starter_code/choices (no solution)
POST /v1/exercises/{id}/submit    grade + record → result
GET  /v1/me/progress              learning path: per-course %, scores, next up
GET  /v1/me/submissions           submission history
GET  /v1/me/files                 DB-backed per-user file catalog (extends /v1/files)
```

**Authoring** (instructor/admin)
```
POST/PUT/DELETE /v1/admin/courses[/{id}]
POST/PUT/DELETE /v1/admin/courses/{id}/lessons[/{id}]
POST/PUT/DELETE /v1/admin/lessons/{id}/exercises[/{id}]
POST            /v1/admin/courses/{id}/publish
GET             /v1/admin/courses/{id}/analytics   pass rates, attempts, time
PUT             /v1/admin/users/{id}/role
```

---

## 6. Phased implementation

- **T0 — DB foundation.** SQLAlchemy async engine/session + `Base`; asyncpg;
  Alembic init; `DATABASE_URL` config; `docker-compose` Postgres service;
  `users` table + **JIT provisioning** (hook the verified identity in
  `require_auth` → upsert `users`, bump `last_seen_at`); `storage.py` adapter
  (`local` default) + `user_files` catalog wired into the existing `/v1/files`;
  `/healthz` DB probe. (S3/MinIO backend can land here or in a later pass.)
- **T1 — Courses (learner read).** `courses`/`lessons`/`exercises` + migrations;
  `GET` browse/detail; `enroll`; seed an intro MATLAB course from fixtures.
- **T2 — Submit + auto-grade.** `submissions`/`progress`; `POST submit` (code via
  matlabc, mcq/short direct); `/v1/me/progress`, `/v1/me/submissions`; scoring,
  completion roll-up.
- **T3 — Authoring CRUD.** instructor/admin course/lesson/exercise CRUD +
  publish; role gating; `users.role` + admin set-role.
- **T4 — Analytics & polish.** per-course/exercise analytics; `tests`-jsonb
  hidden-test runner (function-call assertions); leaderboard; submit rate-limit;
  course import/export (JSON); anti-cheat hardening.

---

## 7. Migrations, seeding, testing, deploy

- **Migrations:** Alembic autogenerate; `alembic upgrade head` on deploy (a
  release step or guarded lifespan hook).
- **Seeding:** `lms/seed.py` loads a starter "Intro to MATLAB" course (lessons +
  graded exercises) from JSON fixtures; idempotent upsert by slug.
- **Testing:** CI gets a **Postgres service container** in `backend.yml`
  (`services: postgres:16`), `DATABASE_URL` → test DB, `alembic upgrade head`,
  per-test transaction rollback for isolation. Grading tests can use the **fake
  matlabc** (deterministic disp/var-store output) for speed, with a heavier lane
  on the real compiler. Local dev: `docker-compose up db`.
- **Deploy (Coolify):** add a Postgres resource → `DATABASE_URL`; run
  migrations on release; enable backups. For `STORAGE_BACKEND=s3`, add a MinIO
  resource (or point at R2/S3) → `S3_ENDPOINT_URL`/`S3_BUCKET`/keys; otherwise
  the `workspace_data` volume is the store (`local`).

---

## 8. Carve-outs / future

Role sync from CyberdyneAuth groups/policies; rich lesson media; cohorts /
assignments / due dates; discussions; plagiarism detection; multi-target
courses (grade python/c/sv via the emitters, not just MATLAB); in-browser IDE;
gamification (badges/streaks/certificates); MCP tools for course/exercise
access (after MCP auth lands).
