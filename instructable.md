Intro

(Picture of setbot)

Ever wanted to beat your friends at the card game Set?
Maybe you'd like to practice against a merciless, trashtalking computer?
Or perhaps you'd just like to learn more about computer vision and deep neural networks?
If so, setbot is the project for you!

Setbot consists of a soft felt playing surface,
a camera, an adjustable stand and a python-based software system that plays the game Set.
When setbot finds a set, it highlights the cards on the screen
and can brag about its skills via a text-to-speech system.

Follow along to learn how the software works and how you can build your own!
Here's how we'll proceed:


1. some background on the game of set
2. the physical setup with a webcam and adjustable stand
3. an overall software gameplan
4. generating training data for the neural network
5. setting up and training a neural network in the cloud
6. how to *recognize* cards with neural networks
7. how to *see* individual cards with computer vision
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
Check the images above for examples!

The NYTimes also posts daily set challenges --
try that out to get the hang of things: http://nytimes.com/set



Step 2: physical setup with a webcam and adjustable stand

We'd like to make a system that can watch the game of set as its being played,
and for this we'll need a webcam.
I picked the Logitech C270 which very easily connects to a computer
and has a pretty good resolution of 1280 x 960 pixels.
It also does nice things like auto-white balance,
though we'll later see that this may not be necessary.

We'll make a simple articulating mount for the webcam
so that it can be positioned over a playing surface.
I also built a small box with a dark felt top to play the game on top of --
the high contrast from the dark background makes card isolation way easier in software.



Step 3: an overall software gameplan

At a high-level, the software will work in three phases:
a data-generation phase, a training phase and an evaluation (gameplay) phase.

*data generation*

Our system will use a neural network to recognize the features in each card,
and we'll use a supervised learning technique to "teach" the network how to correctly identify things.
This involves showing the network a whole lot of card images where we already know the correct answer.
I decided to generate card images in software, rather than take a lot of pictures of individual cards.

1. using SVG and javascript, we will render each of the 81 cards
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
but once they are trained and you have a model, they can be evaluated on new data very quickly.

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
where some deep neural networks have a mindblowing ability
to recognize things from many different orientations.
(See the "VGG" network image, for an example.)
This flexibility comes from their architecture and from the way they're trained.
To learn more about this I'd recommend Andrew Ng's coursera lectures, or Andrei Karpathy's blog.

Neural networks use supervised learning techniques to improve their classification power --
that basically means you need to feed them a lot of labeled data,
and the network will gradually learn to associate these labels with the data.
So we need a lot of data!  How much is a lot?
Well some networks use 100,000,000 images!  I ended up using only about 200,000.
And they all came from mutating a set of 81 "clean" images that I also generated.
I could've taken a whole lot of pictures of set cards, but that's a lot of work.
Having a "fully simulated" input was kind of an interesting test and worked pretty well for this simple example.

Ok, finally, how does it work?
I created each image in SVG on a webpage (see the images for an example)
This was painstaking, especially tracing the "bean" with this drawing tool,
but this left me with a set of 81 clean images.

Then I used an ipython notebook to mutate each image.
Images were randomly rotated, white-balance adjusted and obscured.
This creates some pretty bizarre looking images!
But, interestingly, you'll notice they remain recognizable to a human eye.
And the label of each image is still known because we know the clean image from which it was generated.
Check out the ipython notebook linked here to see the process in more detail.

Whew, so now we have our training data!
We have a lot of obfuscated images and we know the underlying "true value" of the card (the label).
We obscure the images so dramatically so that the network can learn to see pieces of cards --
it also helps us account for small rotation and focus differences that may be present during gameplay.




Step 5: setting up and training a neural network in the cloud

In this step we'll create a convolutional neural network (CNN)
using a machine learning framework called keras.
We'll then use a service called Digital Ocean to setup a cloud server and train the network on our test data.

A CNN is a multilayer network commonly used to recognize images.
There is an input layer that will take in images of each card from the webcam.
This is followed by several convolutional layers, in our case four of them.
Think of each convolution layer as a small window that passes over the layer's input.
The first convolutional layer will see the full input image,
it will pass a small window over the pixels and convert what it sees in that window into a new image.
That new image goes to the next layer which passes another
In this way the image is condensed and CNNs are able to see features at different scales.
For instance, if we were looking at a face,
one layer of a CNN might notice the presence of eyes, a nose and a mouth.
Another layer might see the relationship between two eyes.
A further layer might "study" the shape of an eye.
To read more about convolutional layers, see this post ...

Interspersed between these convolutional sections are activation, pooling and dropout layers.
Activation layers are designed to create a nonlinear response in the network.
One common activation function is the rectified linear unit (the ReLU):
if x > 0, y = x, otherwise y = 0.
Pooling layers ...
Dropout layers are an interesting new development in neural network design.
They simply set a layer's nodes to zero during training with some probability
in the neighborhood of 0.25 to 0.50.
This means the network has to be more flexible during training,
finding different pathways to get to the correct result.

Each of these layers has an associated matrix of weights --
the weight values start off as (almost) random numbers
and the training process constantly tweaks these values
so that the network outputs the correct label for some input training data.
This process is called backpropagation, you can learn more about it here ...

There are many awesome open source tools out there that can manage neural networks for you.
I chose to use keras as it's an easy to understand Python interface.
You can see the code I used to define the neural network here ...

On to training!
I "rented" a Digital Ocean server to train the keras CNN --
this allowed me to use a more powerful computer than the one I have at home.
Digital Ocean also has a nice snapshotting feature so you can work for a bit,
snapshot the state of your machine and destroy the box (and stop paying for it).
Then when you want to resume work you can restore from the snapshot and pick up exactly where you left off.
Setup an account with this link ...
Follow these instructions to setup an Ubuntu box ...
And run this code to setup and train your network ...




Step 5: how to *see* individual cards with computer vision

Our neural network will be setup to recognize individual card images,
but the camera will be taking a single picture of the entire playing field.
We need a method to isolate individual cards from the larger image.

For this we will lean heavily on some OpenCV tools.
Open Computer Vision is a framework that provides a lot of commonly used computer vision methods.
We will use OpenCV methods for color-shifting (cv2.cvtColor),
thresholding (cv2.inRange), and finding contours (cv2.findContours).
You can see this image transformation pipeline in the pictures above.
And you can see the code that makes these adjustments here.
What we're left with is (hopefully) a number of isolated cards
that can be fed into our neural network.



Step 6: how to *recognize* cards with neural networks

This is a bit of an aside as to how our neural network will recognize cards.




7. setting up and training a neural network in the cloud
8. using the trained neural network on your computer
9. ideas for improvements and where to go next!
