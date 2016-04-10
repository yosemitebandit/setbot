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
* an ipython notebook generates more images and data (`generate_input_data.ipynb`):
  * images are cropped,
  * intensified,
  * rotated,
  * resized (shrunk),
  * blurred,
  * and greyscaled
  * the rgb and greyscaled images are converted to separate npy files
* create various models to classify the properties of each card
  * one neural network guesses how many shapes are present in an input image
  (`detect_number.ipynb`) -- 94% accurate
  * another model guesses the color of the shapes (`detect_color.ipynb`) -- 95% accurate
  * another predicts the type shape on the card (`detect_shape.ipynb`) -- 87% accurate
  * and the last predicts the card's texture (`detect_texture.ipynb`) -- 99% accurate
* an evaluation routine loads pre-trained models and examines new pictures of cards
(`evaluator.ipynb`)


#### next steps
* count correct features in evaluator
* make sure the test set has an even distribution of classes (ala CIFAR-10)
* need a larger board -- see sketch
* print stuff on the right sidebar in the camera output
* improve data generation pipeline:
  * adjust colors (via the site?)
  * bring back intensification?
  * consider shrinking the cards
* improvements to the keras test
  * consider going back to four simple, feature-based models if CNN is too slow
  * loading and evaluation in `camera.py`
* consider GPU hosts -- dominodatalab or rescale
* use `np.transpose` like [this post](https://blog.rescale.com/neural-networks-using-keras-on-rescale/)?
* vgg
  * get it [here](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3)
  * set `trainable=False` for the first few, non-Dense layers
  * add a few more layers for my stuff
  * load the weights and train some more..


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
* used [this answer](http://stackoverflow.com/a/12043136/232638) to get opnecv in a virtualenv
