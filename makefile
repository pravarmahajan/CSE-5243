NUM_DOCS=17500
NUM_SHINGLES=5
all:
	python main.py $(NUM_DOCS) $(NUM_SHINGLES)
clean:
	rm *.pyc
	rm data/shingles/*.*
	rm plots/*.*
