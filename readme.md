playing set, the pattern-finding card game


#### pipeline
* generate variants of each card with SVG in the browser
  * URL parameters determine how the cards look, eg ?number=2&color=red&texture=stripes&shape=bean
  * maybe there can be some more parameters later for, like, skewing shapes
  * serve the page from the root via `python -m SimpleHTTPServer` (see `svg-cards`)
* use selenium with the phantomjs driver to iterate through card combinations
and save screenshots (see `download_cards.py`)
* an ipython notebook generates more data by rotating the original set of cards,
and preprocesses the images in other ways (`generate_shapes.ipynb`)
* create various models to classify the properties of each card
  * one neural network guesses how many shapes are present in an input image
  (`count_shapes.ipynb`)
  * another model guesses the color of the shapes (`detect_color.ipynb`)
* an evaluation routine loads pre-trained models and examines new pictures of cards
(`evaluator.ipynb`)


#### next steps
* try more IRL card pictures
* consider adding skewed images


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
