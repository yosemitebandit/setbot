Intro

Picture of setbot
Ever wanted to beat your friends at the card game set?
Maybe you'd like to practice against a merciless, trashtalking computer?
Or perhaps you'd just like to learn more about computer vision and deep neural networks?
If so, setbot is for you!

Setbot consists of a nice felt playing surface,
a camera, an adjustable stand and a python-based software system that plays the game set.
When setbot finds a set, it highlights the cards on the screen
and can brag about its skills via a text-to-speech system.

Follow along to learn how the software works and how you can build your own!
Here's how we'll proceed:

1. background on the game of set
2. physical setup with a webcam and adjustable stand
3. an overall software gameplan
4. generating training data for the neural network
5. how to *see* individual cards with computer vision
6. how to *recognize* cards with neural networks
7. setting up and training a neural network in the cloud
8. using the trained neural network on your computer
9. ideas for improvements and where to go next!

This will primarily focus on some computer vision and neural network concepts,
if you'd just like to explore the code,
it's available [here](https://github.com/yosemitebandit/setbot)!



Step 1: Background on the game of set

Set is made up of 81 unique cards, each with four different features:

* the shape: the card may have diamonds, ovals or "beans" (the squiggly shapes)
* number of shapes: one, two or three
* color: red, green or purple
* and texture: empty, striped or solid

12 cards will be laid in front of you and your job is to find three cards
that either share three out of four features, or have completely different features.
That's a set!
So all but one features must be the same or the cards must be totally different, feature-wise.
Check the images for examples!

The NYTimes also posts daily set challenges --
try that out to get the hang of things: http://nytimes.com/set



Step 2: physical setup with a webcam and adjustable stand

So we'd like to make a system that can watch the game of set as its being played,
for this we'll need a webcam.
I picked the Logitech ... which very easily connects to a computer
and has a pretty good resolution of X x Y pixels.
It also does nice things like auto-white balance,
though we'll later see this may not be that necessary.

We'll make a simple articulating mount for the webcam
so that it can be positioned over a playing surface.
I also built a small box with a dark felt top to play the game on top of --
the high contrast from the dark background makes card isolation way easier in software.



Step 3: an overall software gameplan

Here's how the software works at a high-level,
there are three phases, a data-generation phase, a training phase and an evaluation (gameplay) phase.

*data generation*

Our system will use a neural network to recognize the features in each card,
and we'll use a supervised learning technique to "teach" the network how to correctly identify things.
This involves showing the network a whole lot of card images where we already know the correct answer.
Rather than take a lot of pictures of cards, I decided to generate card images in software.

1. using SVG and javascript, we will render each of the 81 cards on the web
2. we'll use a web scraping system to download each of these images
3. to generate a large body of data, we'll mutate each "clean" image in random ways --
altering its white balance, rotating it and even drastically obscuring the cards with random textures.
This makes the network more robust as it has to learn how to recognize cards despite a good deal of noise.

*training*

At this stage we setup our neural network and begin feeding it our training images --
remember that we know the identity of each card, commonly called its "label."
We'll do this with a powerful server in the cloud to speed things up,
but it can be done at home.
You spend a lot of time upfront to train neural networks,
but once they are trained and you have a model, they can be used very quickly.

1. we'll setup an ubuntu server in the cloud to run our training steps
2. then we'll design the neural network using python
3. and finally we'll start feeding in examples, monitoring the training progress and
capturing the output model once we're satisfied the network is ready

*evaluation (gameplay)*

To actually play the game we'll have the webcam feed data into the pre-trained neural network.

1. a python script runs on my laptop, taking images with the webcam
2. computer vision techniques are used to isolate each individual card in a picture
3. these cards are fed to the neural network for recognition



Step 4: generating training data for the neural network

Setbot could probably work with purely computer vision techniques.
That is, you could write code that discriminates between colors, and shapes and the other features.
Since the features are simple and distinct,
and since the space of possibilities is small (there are only 81 cards total),
this "feature engineering" approach would probably work quite well!

Neural networks take a different approach,
one in which the features of an image don't need to be so clearly demarcated in code.
This has advantages in image recognition,
where some deep neural networks have been shown to have a mindblowing ability
to recognize things from many different orientations (see the VGG image example below).
This flexibility comes from their architecture and from the way they're trained.
To learn more about this I'd recommend [Andrew Ng's coursera lectures](), or [Andrei Karpathy's blog]().

Neural networks use supervised learning techniques to improve their classification power --
that basically means you need to feed them a lot of labeled data,
and the network will gradually learn to associate these labels with the data.
So we need a lot of data!  How much is a lot?
Well some networks use 100M images!  I ended up using only about 100k.
And they all came from mutating a set of 81 "clean" images that I also generated.
I could've taken a whole lot of pictures of set cards, but that's a lot of work.
Having a "fully simulated" input was kind of an interesting test and worked pretty well for this simple example.

Ok, finally, how does it work?
I created each image in SVG on a webpage.
Here's an example:

..

This was painstaking (especially tracing the "bean" with [this drawing tool](..))
but got left me with a set of 81 clean images.

Then I used an ipython notebook to mutate each image.
Images were randomly rotated, white-balance adjusted and obscured.
This creates some pretty bizarre looking images!
But, interestingly, you'll notice they remain recognizable to a human eye.
And the label of each image is still known because we know the clean image from which it was generated.
Check out the ipython notebook to see the process in more detail.

So now we have our training data!
We have a lot of obfuscated images and we know the underlying "true value" of the card (the label).
We obscure the images so dramatically so that the network can learn to see pieces of cards --
it also helps us account for small rotation and focus differences that may be present during gameplay.



Step 5: how to *see* individual cards with computer vision

Our neural network will be setup to recognize individual card images,
but the camera will be taking a single picture of the entire playing field.
We need a method to isolate individual cards from the larger image.

For this we will lean heavily on some OpenCV tools.
Open Computer Vision is a framework that provides a lot of commonly used computer vision methods.



6. how to *recognize* cards with neural networks
7. setting up and training a neural network in the cloud
8. using the trained neural network on your computer
9. ideas for improvements and where to go next!
