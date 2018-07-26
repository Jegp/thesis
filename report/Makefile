SOURCE_IN := report.tex
SOURCE_OUT := report.pdf
BIB_SOURCE := bib.bib

DOTFILE_DIRS := chapters/volr

.PHONY: all clean $(DOTFILE_DIRS)
all: build

build: $(DOTFILE_DIRS)
	rm report.pdf -f
	latexmk -pdf ${SOURCE_IN}

$(DOTFILE_DIRS): 
	$(foreach f,$(wildcard $@/*.dot), \
		dot2tex $f --figonly -f tikz > $(basename $f).tex \
	) 

clean:
	@rm -f *.log *.aux *.bbl *.bcf *.blg *.fdb_latexmk *.fls *.glg *.glo *.gls *.ist *.out *.run.xml *.synctex.gz ${SOURCE_OUT}

watch:
	while inotifywait -e modify -r chapters/*.tex ; do $(MAKE) ; done