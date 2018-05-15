SOURCE_IN = report.tex
SOURCE_OUT = report.pdf
BIB_SOURCE = bib.bib

all: build

build:
	latexmk -pdf ${SOURCE_IN}

preview:
	atom ${SOURCE_OUT}

clean:
	@rm -f *.log *.aux *.bbl *.bcf *.blg *.fdb_latexmk *.fls *.glg *.glo *.gls *.ist *.out *.run.xml *.synctex.gz ${SOURCE_OUT}
