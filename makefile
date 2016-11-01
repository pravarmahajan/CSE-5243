all:
	python -W main.py
clean:
	rm *.pyc
	rm data/output/*.*
	rm plots/*.*
preprocessing:
	python preprocessing.py
classifier:
	python -W ignore classifier.py
