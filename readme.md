playing set, the pattern-finding card game

A webcam + OpenCV isolates cards and feeds them to a CNN for identification.
The rules of set are pretty straightforward, so once cards are identified it
doesn't take long to find sets of three.

![gameplay](play.png)


#### pipeline
* generate variants of each card with SVG in the browser
  * URL parameters determine how the cards look, eg `?number=2&color=red&texture=stripes&shape=bean`
  * maybe there can be some more parameters later for, like, skewing shapes
  * serve the page from the root via `python -m SimpleHTTPServer` (see `svg-cards`)
* use selenium with the phantomjs driver to iterate through card combinations
and save screenshots (see `download_cards.py`)
* an ipython notebook generates more images and data (`generate_input_data.ipynb`):
  * images are cropped,
  * resized,
  * rotated,
  * obfuscated,
  * their white balance is shifted,
  * and blurred
  * the rgb images are converted to separate npy data files
* a keras CNN trains on ~200k cards --
typically for about 8hrs on an 8 core digital ocean box to get >90% accuracy
(see `cnn_with_generator.py`) -- I haven't yet tried a GPU
* `camera.py` finds cards via thresholding
and sends them through the trained model to understand their characteristics,
then the `set_the_game.py` module helps identify a set


#### old notebooks
* I created various models to classify the properties of each card
  * one neural network guesses how many shapes are present in an input image
  (`detect_number.ipynb`) -- 94% accurate
  * another model guesses the color of the shapes (`detect_color.ipynb`) -- 95% accurate
  * another predicts the type shape on the card (`detect_shape.ipynb`) -- 87% accurate
  * and the last predicts the card's texture (`detect_texture.ipynb`) -- 99% accurate
* an evaluation routine loads pre-trained models and examines new pictures of cards
(`evaluator.ipynb`)
* ..this didn't work all that well in practice though,
I think the synthetic data I used for training wasn't representative --
I may have also been evaluating my model incorrectly, I had issues with that
later in the 81 class work


#### next steps
* need a larger board -- see sketch
* print stuff on the right sidebar in the camera output
* handle card prediction dupes
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
* something's not right with keras -- `batch_size = 100` and `samples_per_epoch = 1000`
does not converge in `cnn_with_generator` but if I use 10x more `samples_per_epoch`
I do get convergence.. should I go back to vanilla tensorflow?


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
* I thought I would need `setbot-server` to do evaluation in the cloud,
but that's not the case -- keeping it around as it's a good flask demo
