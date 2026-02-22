import os
import uvicorn

if __name__ == "__main__":
    reload_enabled = os.getenv("UVICORN_RELOAD", "false").strip().lower() == "true"
    host = host = "0.0.0.0"  
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("gpt_project.api:app", host=host, port=port, reload=reload_enabled)
