# Cortex Engine Deployment

## Goal
Keep the app online 24/7 and keep live data (news/stocks) updated even when your laptop is off.

## Local commands
- Run API:
  - `python run_api.py`
- Run background updater:
  - `python run_worker.py`
- Run smoke test:
  - `powershell -ExecutionPolicy Bypass -File .\scripts\smoke_test.ps1`
  - optional model path: `powershell -ExecutionPolicy Bypass -File .\scripts\smoke_test.ps1 -IncludeChat`

## Environment variables
- `OPENAI_API_KEY`: required.
- `HOST`: API bind host (`0.0.0.0` in cloud).
- `PORT`: API port.
- `UVICORN_RELOAD`: `false` for production.
- `LIVE_UPDATE_ENABLED`: `true`/`false`.
- `LIVE_UPDATE_INTERVAL_SECONDS`: refresh interval (default `300`).
- `STOCK_SYMBOLS`: comma-separated stock list, e.g. `AAPL,MSFT,SPY`.
- `NEWS_FEED_URLS`: optional comma-separated RSS feeds.
- `WEB_SEARCH_PROVIDER`: use `multi` for fallback search providers.
- `BRAVE_SEARCH_API_KEY`: optional (improves live web coverage when set).
- `DB_PATH`: shared database file path if both API and worker run on same machine.
- `CREATOR_EMAIL`: your creator/admin email for unlimited access.
- `STRIPE_SECRET_KEY`: Stripe secret API key (`sk_live_...` in production).
- `STRIPE_WEBHOOK_SECRET`: webhook signing secret for `/billing/webhook`.
- `STRIPE_PRICE_PRO5_ID`: Stripe Price ID for your $5 plan.
- `STRIPE_PRICE_PRO10_ID`: Stripe Price ID for your $10 plan.
- `APP_BASE_URL`: public app URL, used for Stripe success/cancel redirects.

## Secret safety
- Never commit `.env` or API keys to git.
- Rotate keys immediately if they appear in terminal screenshots or logs.

## Render deployment
This repo includes `render.yaml` with:
- `trent-gpt-web` (FastAPI web service with embedded updater enabled)

## Important production note
SQLite is fine for local/single-machine deploys. For multi-service cloud deployments, use a shared database (Postgres) instead of local SQLite files.

If separate web + worker containers run without a shared database, they will not share one SQLite file reliably.
The default `render.yaml` avoids this by running updates inside the web service process (`RUN_UPDATER_IN_API=true`).

## Billing setup (Stripe)
- Checkout endpoint: `POST /billing/checkout` (requires login).
- Webhook endpoint: `POST /billing/webhook` (configure in Stripe Dashboard).
- Recommended webhook events:
  - `checkout.session.completed`
  - `customer.subscription.created`
  - `customer.subscription.updated`
- Bank payouts are configured in Stripe Dashboard:
  - `Settings -> Business -> Bank accounts and scheduling`.

## Recommended production architecture
- Web service: FastAPI (`run_api.py`)
- Optional dedicated worker: updater (`run_worker.py`) only if using shared DB
- Shared data store: Postgres
- Optional cache: Redis
