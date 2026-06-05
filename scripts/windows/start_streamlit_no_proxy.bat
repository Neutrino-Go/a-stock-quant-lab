@echo off
chcp 65001 >nul
setlocal
set "PYTHONIOENCODING=utf-8"

python -c "print('\u003d'*40); print('Stocking Lab / A-Stock Quant Lab'); print('\u65e0\u4ee3\u7406\u6a21\u5f0f\u542f\u52a8 Streamlit'); print('\u003d'*40)"
echo.

cd /d "C:\Users\18961\Documents\Stock Trading"

python -c "print('\u5f53\u524d\u9879\u76ee\u76ee\u5f55\uff1a')"
cd
echo.

python -c "print('\u6b63\u5728\u4e34\u65f6\u6e05\u9664\u5f53\u524d CMD \u4f1a\u8bdd\u91cc\u7684\u4ee3\u7406\u53d8\u91cf...')"
set "HTTP_PROXY="
set "HTTPS_PROXY="
set "ALL_PROXY="
set "http_proxy="
set "https_proxy="
set "all_proxy="
set "NO_PROXY=*"
set "no_proxy=*"
echo.

python -c "print('\u5f53\u524d\u4ee3\u7406\u8bca\u65ad\uff1a')"
python -c "from utils.network import get_proxy_diagnostics; print(get_proxy_diagnostics())"
echo.

python -c "print('\u6b63\u5728\u6d4b\u8bd5\u4e1c\u65b9\u8d22\u5bcc\u63a5\u53e3...')"
python -c "from utils.network import test_eastmoney_connectivity; result=test_eastmoney_connectivity(); print(result); raise SystemExit(0 if result.get('ok') else (2 if result.get('error_type') == 'ProxyError' else 1))"
set "NETWORK_RC=%ERRORLEVEL%"
echo.

if "%NETWORK_RC%"=="2" (
    python -c "print('\u4f60\u7684\u7cfb\u7edf\u6216\u4ee3\u7406\u8f6f\u4ef6\u4ecd\u5728\u5f3a\u5236\u4ee3\u7406\uff0c\u8bf7\u5173\u95ed VPN/\u4ee3\u7406\u8f6f\u4ef6\uff0c\u6216\u53bb Windows \u8bbe\u7f6e\u91cc\u5173\u95ed\u4ee3\u7406\u3002')"
    python -c "print('\u4ecd\u4f1a\u7ee7\u7eed\u542f\u52a8 Streamlit\uff0c\u4f46\u66f4\u65b0\u80a1\u7968\u6570\u636e\u53ef\u80fd\u7ee7\u7eed\u5931\u8d25\u3002')"
) else if "%NETWORK_RC%"=="0" (
    python -c "print('\u7f51\u7edc\u5df2\u80fd\u8bbf\u95ee\u4e1c\u65b9\u8d22\u5bcc\uff0c\u53ef\u4ee5\u7ee7\u7eed\u542f\u52a8 Streamlit\u3002')"
) else (
    python -c "print('\u7f51\u7edc\u6d4b\u8bd5\u672a\u901a\u8fc7\u3002\u53ef\u80fd\u662f\u5f53\u524d\u7f51\u7edc\u65e0\u6cd5\u8bbf\u95ee\u4e1c\u65b9\u8d22\u5bcc\uff0c\u6216\u4e1c\u65b9\u8d22\u5bcc\u4e34\u65f6\u4e0d\u53ef\u7528\u3002')"
    python -c "print('\u4ecd\u4f1a\u7ee7\u7eed\u542f\u52a8 Streamlit\uff0c\u4f60\u53ef\u4ee5\u5148\u6d4b\u8bd5\u672c\u5730\u5df2\u6709\u6570\u636e\u548c\u9875\u9762\u7a7a\u72b6\u6001\u3002')"
)

echo.
python -c "print('\u6b63\u5728\u542f\u52a8 Streamlit...')"
echo http://localhost:8501
echo.
streamlit run app.py

echo.
python -c "print('Streamlit \u5df2\u9000\u51fa\u3002\u6309\u4efb\u610f\u952e\u5173\u95ed\u7a97\u53e3...')"
pause >nul
