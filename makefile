all:
	python main.py
clean:
	rm *.pyc
	rm data/output/*.*
	rm plots/*.*
classifier:
	python -W ignore classifier.py
