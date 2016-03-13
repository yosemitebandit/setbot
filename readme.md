playing set, the pattern-finding card game


#### pipeline
* generate variants of each card with SVG in the browser
  * seems like snap-svg is the best tool now, the raphaeljs dev works on that project now..
  * URL parameters determine how the cards look, eg ?number=2&color=red&texture=stripes&shape=bean
  * maybe there can be some more parameters later for, like, skewing shapes
* serve the page from the root via `python -m SimpleHTTPServer`
* use selenium with the phantomjs driver to iterate through card combinations
and save screenshots
* an ipython notebook generates more data by rotating the original set of cards,
and preprocesses the images in other ways (`generate_shapes.ipynb`)
* one neural network guesses how many shapes are present in an input image
(`count_shapes.ipynb`)


#### next steps
* classifier that uses RGB data (picking out the color)
* save the trained model and use it in an evaluation script
* end-to-end test with an actual picture of a card


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
