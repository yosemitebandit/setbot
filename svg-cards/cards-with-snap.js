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

var bean = paper.g(topHalfBean, bottomHalfBean);
bean.attr({
  fill: 'white',
  stroke: 'red',
  strokeWidth: 3,
});
