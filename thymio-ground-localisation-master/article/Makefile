name=thymio-localisation-dars
targetpdf=$(name).pdf
source=$(name).tex
bib=$(name).bib
svgfigs=$(wildcard figures/*.svg)
rfigs=$(wildcard figures/*.r)
genfigs=$(svgfigs:figures/%.svg=generated-figures/%.pdf) $(rfigs:figures/%.r=generated-figures/%.pdf)
deps=$(wildcard *.tex) $(wildcard figures/*.pdf) $(wildcard figures/*.jpg) $(wildcard figures/*.png) $(genfigs) Makefile


%.pdf: %.tex $(bib) $(deps) $(genfigs) create-working-dirs 
	cd build && TEXINPUTS=..:${TEXINPUTS} pdflatex ../$<
	cd build && BIBINPUTS=..:${BIBINPUTS} bibtex $(patsubst %.pdf,%,$@)
	cd build && TEXINPUTS=..:${TEXINPUTS} pdflatex ../$<
	cd build && TEXINPUTS=..:${TEXINPUTS} pdflatex ../$<
	mv build/$@ .

all: $(targetpdf)

create-working-dirs:
	mkdir -p build generated-figures

generated-figures/%.pdf: figures/%.svg
	inkscape --export-area-page --export-pdf=$@ -f=$<

generated-figures/%.pdf: figures/%.r
	cd build && R BATCH --no-save < ../$< && mv $(patsubst generated-figures/%.pdf,%.pdf,$@) ../generated-figures/

%-emb.pdf: %.pdf
	gs -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -dEmbedAllFonts=true  -dPDFSETTINGS=/prepress -dSubsetFonts=true -sOutputFile=$@ -f $<

clean: buildclean

buildclean:
	rm -rf build */*.Rout */.*.Rout */.RData *~ figures/*~ generated-figures/*

distclean: clean
	rm -f $(targetpdf)
	rm -rf generated-figures
