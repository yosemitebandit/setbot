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
var cardWidth = aspectRatio * cardHeight;  // 344
var cardRoundedness = 20;
var cardBackground = paper.rect(0, 0, cardWidth, cardHeight, cardRoundedness);
cardBackground.attr({
  fill: '#fff',
  stroke: '#000',
});


// one green empty oval
var ovalWidth = 250;
var ovalHeight = 100;
var ovalRoundedness = ovalHeight / 2;
var oval = paper.rect(45, 220, ovalWidth, ovalHeight, ovalRoundedness);
oval.attr({
  fill: '#fff',
  stroke: '#35bd2d',
  strokeWidth: '6',
});


/*
// two red striped beans
var bean = paper.path(
  'M 29, 104 ' +
  'C 22 140, 29 159, 37 172 ' +
  'C 49 191, 70 190, 93 172 ' +
  'C 124 148, 165 154, 178 162' +
  'C 212 182, 261 178, 276 174' +
  'C 310 165, 331 137, 338 124' +
  'C 352 98, 351 57, 339 45 ' +
  'C 316 22, 288 58, 266 63 ' +
  'C 246 67, 228 72, 205 63 ' +
  'C 181 54, 137 39, 122 38 ' +
  'C 89 36, 69 50, 57 59' +
  'C 43 69, 37 84, 30 102 ' +
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
*/


/*
// three purple solid diamonds
var diamond = paper.path(
  'M 175 300 ' +
  'l 120 -55 ' +
  'l 120 55 ' +
  'l -120 55 ' +
  'l -120 -55 ' +
  'z'
);
var diamondMatrix = new Snap.Matrix();
diamondMatrix.translate(130, -170);
diamond.transform(diamondMatrix);

diamond.attr({
  stroke: 'purple',
  fill: 'purple',
});

var secondDiamond = diamond.clone();
var secondDiamondMatrix = new Snap.Matrix();
secondDiamondMatrix.translate(130, -10);
secondDiamond.transform(secondDiamondMatrix);

var thirdDiamond = diamond.clone();
var thirdDiamondMatrix = new Snap.Matrix();
thirdDiamondMatrix.translate(130, 150);
thirdDiamond.transform(thirdDiamondMatrix);
*/
