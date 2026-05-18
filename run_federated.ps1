# run_federated.ps1
Write-Host "--- Federe Ogrenme Simülasyonu Baslatiliyor ---" -ForegroundColor Cyan

# 1. Sunucuyu arka planda baslat
Write-Host "[Server] Baslatiliyor..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "python src/server.py" -WindowStyle Normal
Start-Sleep -Seconds 5 # Sunucunun ayaga kalkmasi icin bekle

# 2. 2 adet istemciyi (hastane) sirayla baslat
for ($i = 1; $i -le 2; $i++) {
    Write-Host "[Client $i] Baslatiliyor..." -ForegroundColor Green
    Start-Process powershell -ArgumentList "python src/client.py --client_id $i" -WindowStyle Normal
    Start-Sleep -Seconds 2 # VRAM'in sismesini onlemek icin kademeli baslat
}

Write-Host "Tum sistemler devrede! Egitimi pencerelerden takip edebilirsin." -ForegroundColor Magenta