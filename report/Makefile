SOURCE_IN := report.tex
SOURCE_OUT := report.pdf
BIB_SOURCE := bib.bib
DOTFILE_DIRS := chapters/volr

graphs: DOT = dot

.PHONY: all clean graphs $(DOTFILE_DIRS)
all: build

build: 
	rm report.pdf -f
	latexmk -pdf ${SOURCE_IN}

graphs: $(DOTFILE_DIRS)

$(DOTFILE_DIRS): 
	$(foreach f,$(wildcard $@/*.dot), \
		$(DOT) -Efontname="Latin Modern Math" -Tpdf $f > $(basename $f).pdf; \
	) 

clean:
	@rm -f *.log *.aux *.bbl *.bcf *.blg *.fdb_latexmk *.fls *.glg *.glo *.gls *.ist *.out *.run.xml *.synctex.gz ${SOURCE_OUT}

watch:
	while inotifywait -e modify -r chapters/**/*.tex ; do $(MAKE) ; done