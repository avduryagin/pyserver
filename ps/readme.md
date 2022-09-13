## Установить [python-3.8.7]
(https://www.python.org/downloads/release/python-387/)

## Создание и активация venv
1. py -3.8 -m venv .venv
2. Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
3. .\\.venv\Scripts\activate

 ## Установить пакеты
 * pip install -r requirements\dev.txt 
 
 ## Запуск waitress сервера 
* python server.py
* Запускается по F5 в VSCode