import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

# Dirección interna de la API real (main_api.py)
API_REAL = "http://127.0.0.1:8000"

# Sin documentación pública: el navegador solo conoce /api
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


async def reenviar(metodo: str, ruta: str, cuerpo=None) -> JSONResponse:
    try:
        async with httpx.AsyncClient(timeout=10) as cliente:
            r = await cliente.request(metodo, f"{API_REAL}{ruta}", json=cuerpo)
        return JSONResponse(r.json(), status_code=r.status_code)
    except httpx.RequestError:
        return JSONResponse({"detail": "Servicio de predicción no disponible."}, status_code=503)


@app.get("/")
def inicio():
    return FileResponse("index.html")


@app.get("/style.css")
def estilos():
    return FileResponse("style.css", media_type="text/css")


@app.get("/app.js")
def script():
    return FileResponse("app.js", media_type="application/javascript")


@app.post("/api")
async def api(request: Request):
    try:
        cuerpo = await request.json()
    except Exception:
        return JSONResponse({"detail": "JSON inválido."}, status_code=400)
    return await reenviar("POST", "/predict", cuerpo)


@app.get("/api/posiciones")
async def api_posiciones():
    return await reenviar("GET", "/posiciones")