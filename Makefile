train:
	python training/train.py

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/ -v

docker:
	docker build -t credit-risk-api . && docker run -p 8000:8000 credit-risk-api
