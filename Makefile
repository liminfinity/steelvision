APP_PORT=8501
NGROK_BIN=ngrok

.PHONY: run tunnel all stop

# Запуск Streamlit-приложения
run:
	streamlit run app.py --server.port $(APP_PORT) --server.address 0.0.0.0

# Проброс порта через ngrok
tunnel:
	$(NGROK_BIN) http $(APP_PORT)

# Запустить и приложение, и туннель параллельно
all:
	$(MAKE) -j 2 run tunnel

# Остановить все процессы streamlit
stop:
	pkill -f "streamlit run app.py" || true