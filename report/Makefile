SOURCE_IN = report.tex
SOURCE_OUT = report.pdf
BIB_SOURCE = bib.bib

all: clean build

build:
	rm report.pdf -f
	latexmk -pdf ${SOURCE_IN}

clean:
	@rm -f *.log *.aux *.bbl *.bcf *.blg *.fdb_latexmk *.fls *.glg *.glo *.gls *.ist *.out *.run.xml *.synctex.gz ${SOURCE_OUT}
