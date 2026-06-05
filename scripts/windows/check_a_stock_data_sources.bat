@echo off
chcp 65001 >nul
setlocal
set "PYTHONIOENCODING=utf-8"

python -c "print('\u003d'*48); print('A-share daily data source diagnostics'); print('\u003d'*48)"
echo.

cd /d "C:\Users\18961\Documents\Stock Trading"

set "HTTP_PROXY="
set "HTTPS_PROXY="
set "ALL_PROXY="
set "http_proxy="
set "https_proxy="
set "all_proxy="
set "NO_PROXY=*"
set "no_proxy=*"

python scripts\check_a_stock_data_sources.py

echo.
python -c "print('\u8bca\u65ad\u5b8c\u6210\u3002\u6309\u4efb\u610f\u952e\u5173\u95ed\u7a97\u53e3...')"
pause >nul
