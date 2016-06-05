.PHONY: all clean

SRC=$(wildcard *.hs)

all: $(SRC) build
	ghc --make -outputdir build $(SRC)

build:
	mkdir -p build

clean:
	rm build/*.o build/*.hi

