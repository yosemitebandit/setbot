// setup
var width = 800;
var height = 600;
var paper = Snap(width, height);


// background
var background = paper.rect(0, 0, width, height);
background.attr('fill-opacity', 0.0);


// card
var cardAspectRatio = 0.64,
    cardHeight = 550,
    cardWidth = cardAspectRatio * cardHeight,  // 344
    cardCenter = [cardWidth / 2, cardHeight / 2],
    cardRoundedness = cardHeight / 28,  // ~20
    cardBackground = paper.rect(0, 0, cardWidth, cardHeight, cardRoundedness);

cardBackground.attr({
  fill: '#fff',
  stroke: '#000',
});


// shape parameters
var cardWidthToShapeWidthRatio = 1.477,
    shapeWidth = cardWidth / cardWidthToShapeWidthRatio,
    shapeAspectRatio = 2.133,
    shapeHeight = shapeWidth / shapeAspectRatio;


// one green empty oval
/*
var ovalWidth = shapeWidth,
    ovalHeight = shapeHeight,
    ovalTopLeftCorner = [cardCenter[0] - ovalWidth / 2, cardCenter[1] - ovalHeight / 2],
    ovalRoundedness = ovalHeight / 2,
    oval = paper.rect(ovalTopLeftCorner[0], ovalTopLeftCorner[1], ovalWidth, ovalHeight, ovalRoundedness);
oval.attr({
  fill: '#fff',
  stroke: '#35bd2d',
  strokeWidth: '6',
});
*/


// two red striped beans
var bean = paper.path(
  'M 29, 104 ' +
  'c -7  36, 0  55,  8  68 ' +
  'c 12  19, 33 18,  56 0 ' +
  'c 31 -24, 72 -18, 85 -10 ' +
  'c 34 20, 83 16, 98 12 ' + // 276 174
  'C 310 165, 331 137, 338 124 ' + // 338 124
  'C 352 98,  351 57,  339 45 ' + // 339 45
  'C 316 22,  288 58,  266 63 ' + // 266 63
  'C 246 67,  228 72,  205 63 ' + // 205 63
  'C 181 54,  137 39,  122 38 ' + // 122 38
  'C 89  36,  69  50,  57  59 ' +  // 57 59
  'C 43  69,  37  84,  30  102 ' + // 30 102
  'z'
);
var beanScaleFactor = 0.8;
var beanMatrix = new Snap.Matrix()
beanMatrix.scale(beanScaleFactor);
beanMatrix.translate(340, 160);
bean.transform(beanMatrix);

// create the pattern
var patternPath = (
  'M 5 -10 ' +
  'L 5 10 '
);
var stripedPattern = paper.path(patternPath).attr({
  fill: 'none',
  stroke: 'red',
  strokeWidth: 1,
}).toPattern(0, 0, 8, 10);

bean.attr({
  fill: stripedPattern,
  stroke: 'red',
  strokeWidth: 6,
});

// make a second bean and move it in place
var secondBean = bean.clone();
secondBeanMatrix = new Snap.Matrix();
secondBeanMatrix.scale(beanScaleFactor);
secondBeanMatrix.translate(340, 360);
secondBean.transform(secondBeanMatrix);


/*
// three purple solid diamonds
var leftDiamondCorner = [cardCenter[0] - shapeWidth / 2, cardCenter[1]],
    diamond = paper.path(
      'M ' + leftDiamondCorner[0] + ' ' + leftDiamondCorner[1] + ' ' +
      'l 120 -55 ' +
      'l 120 55 ' +
      'l -120 55 ' +
      'l -120 -55 ' +
      'z'
    );

diamond.attr({
  stroke: 'purple',
  fill: 'purple',
});

var cardHeightToVerticalShapeDisplacementRatio = 3.488,
    verticalShapeDisplacement = cardHeight / cardHeightToVerticalShapeDisplacementRatio,
    secondDiamond = diamond.clone(),
    secondDiamondMatrix = new Snap.Matrix();

secondDiamondMatrix.translate(0, -verticalShapeDisplacement);
secondDiamond.transform(secondDiamondMatrix);

var thirdDiamond = diamond.clone(),
    thirdDiamondMatrix = new Snap.Matrix();

thirdDiamondMatrix.translate(0, verticalShapeDisplacement);
thirdDiamond.transform(thirdDiamondMatrix);
*/
