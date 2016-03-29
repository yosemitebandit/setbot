playing set, the pattern-finding card game

I'm generating images of each card via SVG
and then using several TensorFlow models to classify these synthetic examples.


#### pipeline
* generate variants of each card with SVG in the browser
  * URL parameters determine how the cards look, eg `?number=2&color=red&texture=stripes&shape=bean`
  * maybe there can be some more parameters later for, like, skewing shapes
  * serve the page from the root via `python -m SimpleHTTPServer` (see `svg-cards`)
* use selenium with the phantomjs driver to iterate through card combinations
and save screenshots (see `download_cards.py`)
* an ipython notebook generates more data by rotating the original set of cards,
preprocesses the images in other ways, and generates data files (`generate_shapes.ipynb`)
* create various models to classify the properties of each card
  * one neural network guesses how many shapes are present in an input image
  (`detect_number.ipynb`) -- 99% accurate
  * another model guesses the color of the shapes (`detect_color.ipynb`) -- 79% accurate
  * another predicts the type shape on the card (`detect_shape.ipynb`) -- 92% accurate
  * and the last predicts the card's texture (`detect_texture.ipynb`) -- 94% accurate
* an evaluation routine loads pre-trained models and examines new pictures of cards
(`evaluator.ipynb`)


#### next steps
* consider adding skewed images
* use [more cores for image generation](http://stackoverflow.com/a/23537302/232638)


#### rules of set
* each card in set has four characteristics:
color, symbol, number of symbols and texture.
* color variants: red, green or purple
* symbol variants: diamond, squiggle or oval
* number variants: one, two or three
* texture variants: solid, open or striped
* there are 81 cards in the deck, one with each unique variation
* 12 cards are dealt initially
* players attempt to identify "sets" --
three cards that can be categorized as "two of `X` and one of `Y`" do /not/ constitute a set
* if no sets are found, three additional cards are dealt
* this continues until there are no more cards to deal,
at this point, the player with the most sets wins


#### misc
* useful [bezier editor](http://www.victoriakirst.com/beziertool)
* [isolated bean image for tracing](http://i.imgur.com/U9k6OMR.png)
