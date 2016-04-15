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
* need a larger board -- see sketch
* print stuff on the right sidebar in the camera output
* improve data generation pipeline:
  * adjust colors (via the site?)
  * bring back intensification?
  * consider shrinking the cards
* consider GPU hosts -- dominodatalab or rescale or just AWS
* vgg
  * get it [here](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3)
  * set `trainable=False` for the first few, non-Dense layers
  * add a few more layers for my stuff
  * load the weights and train some more..
* train on rotated / mis-scaled cards or fix the input rotation issue --
seems fixed if each dealt card has a slight ccw rotation..
* consider training on larger images (having issue with color / shading sometimes) --
struggles with 1 striped diamond vs three
* on batching -- could try your generator examples for longer..
could also just run `model.fit` with self-made batches
* something's not right -- `batch_size = 100` and `samples_per_epoch = 1000`
does not converge in `cnn_with_generator` but if I use 10x more `samples_per_epoch`
I do get convergence..should I go back to vanilla tensorflow?


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
