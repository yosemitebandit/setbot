// setup
var width = 800;
var height = 600;
var paper = Snap(width, height);

// background
var background = paper.rect(0, 0, width, height);
background.attr('fill-opacity', 0.0);

// card
var aspectRatio = 0.625;
var cardHeight = 550;
var cardWidth = aspectRatio * cardHeight;
var cardRoundedness = 20;
var cardBackground = paper.rect(250, 25, cardWidth, cardHeight, cardRoundedness);
cardBackground.attr({
  fill: '#fff',
  stroke: '#000',
});

// two red striped beans
var topHalfBean = paper.path(
  'M 280 300 ' +
  'c 0 0, 10 -60, 50 -60 ' +
  'c 40 0, 85 20, 125 20 ' +
  'c 95 0, 140 -70, 110 40 '
);
var beanScaleFactor = 0.8;
var topMatrix = new Snap.Matrix();
topMatrix.scale(beanScaleFactor);
topMatrix.translate(100, 0);
topHalfBean.transform(topMatrix);

var bottomHalfBean = topHalfBean.clone();
var bottomMatrix = new Snap.Matrix();
bottomMatrix.scale(beanScaleFactor);
bottomMatrix.rotate(180, 400, 300);
bottomMatrix.translate(-145, 0);
bottomHalfBean.transform(bottomMatrix);

// create the pattern
var patternPath = (
  'M 10 -10 ' +
  'L 10 10 '
);
// magic numbers for good alignment with 10,-10 -> 10,10: 11 and 15
var inversePatternDensity = 11;
var stripedPattern = paper.path(patternPath).attr({
  fill: 'none',
  stroke: 'red',
  strokeWidth: 1,
}).toPattern(0, 0, inversePatternDensity, 10);

// group the top and bottom half together
var bean = paper.g(topHalfBean, bottomHalfBean);
bean.attr({
  fill: stripedPattern,
  stroke: 'red',
  strokeWidth: 3,
});

// move the whole thing
var beanMatrix = new Snap.Matrix();
beanMatrix.translate(0, -30);
bean.transform(beanMatrix);

// make a second bean and move it in place
var secondBean = bean.clone();
secondBeanMatrix = new Snap.Matrix();
secondBeanMatrix.translate(0, 120);
secondBean.transform(secondBeanMatrix);
