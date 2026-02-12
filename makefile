.PHONY: setup download-data preprocess features train predict evaluate clean all
setup:
	pip install -r requirements.txt
download-data:
	python src/download.py
preprocess:
	python src/preprocess.py
features:
	python src/features.py
train:
	python src/train.py
predict:
	python src/predict.py
evaluate:
	python src/evaluate.py

clean:
	rm -rf data/processed/*
	rm -rf features/*
	rm -rf models/*
	rm -rf results/*
all: setup download-data preprocess features train predict evaluate
