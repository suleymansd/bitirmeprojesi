# Railway + Vercel Canli Yayin

## 1) Railway (Backend + Model API)

1. Railway'de yeni proje olustur ve bu repoyu bagla.
2. Root dizin: `cilt-kanseri-teshisi-bitirme`
3. Deploy olduktan sonra `https://<railway-domain>/api/health` kontrol et.
4. Model secimi icin opsiyonel env:
   - `MODEL_MODE=federated` veya `baseline`
   - `FEDERATED_ROUND=12` (ornek)

Not: `Procfile` ve `railway.json` mevcut, start komutu otomatik:
`uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}`

## 2) Vercel (Frontend)

1. Vercel'de bu repoyu import et.
2. Root dizin yine: `cilt-kanseri-teshisi-bitirme`
3. Deploy'dan once `vercel.json` icindeki asagidaki satiri guncelle:

`https://REPLACE_WITH_YOUR_RAILWAY_DOMAIN/api/:path*`

ornek:

`https://dermai-production.up.railway.app/api/:path*`

4. Deploy et.

## 3) Son Kontrol

- Vercel URL acildiginda dashboard gelmeli.
- Frontend'den analiz yapinca istekler Vercel `/api/*` uzerinden Railway'e proxy edilir.
- Sunumda tek URL olarak Vercel URL kullanabilirsin.
