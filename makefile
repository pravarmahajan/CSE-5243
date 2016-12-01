all:
	python cba_classifier.py
clean:
	rm *.pyc
	rm data/shingles/*.*
	rm plots/*.*
