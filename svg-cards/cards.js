// Setup the drawing environment.
var width = 800,
    height = 600,
    paper = Snap(width, height);


// Setup the SVG background.
var background = paper.rect(0, 0, width, height).attr({
  fillOpacity: 0.0
});


// Make the card background.
var cardAspectRatio = 0.64,
    cardHeight = 550,
    cardWidth = cardAspectRatio * cardHeight,  // 344
    cardCenter = [cardWidth / 2, cardHeight / 2],
    cardRoundedness = cardHeight / 28;  // 20

var cardBackground = paper.rect(0, 0, cardWidth, cardHeight, cardRoundedness).attr({
  fill: '#fff',
  stroke: '#000',
});


// Variables
var inputNumberOfShapes = 2,
    inputColor = 'red',
    // green '#35bd2d'
    inputTexture = 'solid',
    inputShape = 'diamond';


// Shape parameters.
var cardWidthToShapeWidthRatio = 1.477,
    shapeWidth = cardWidth / cardWidthToShapeWidthRatio,
    shapeAspectRatio = 2.133,
    shapeHeight = shapeWidth / shapeAspectRatio;


// Set the locations of the 2- and 3-shape center points.
// The 1-shape point is just the center, as is the 3-shape midpoint.
var cardHeightToTwoShapeOffsetRatio = 3.524,
    cardHeightToThreeShapeOffsetRatio = 3.488,
    twoShapeHighPointOffset = -cardHeight / cardHeightToTwoShapeOffsetRatio / 2,
    twoShapeLowPointOffset = cardHeight / cardHeightToTwoShapeOffsetRatio / 2,
    threeShapeHighPointOffset = -cardHeight / cardHeightToThreeShapeOffsetRatio,
    threeShapeLowPointOffset = cardHeight / cardHeightToThreeShapeOffsetRatio;


// Setup texture.
switch (inputTexture) {
  case 'solid':
    var fillTexture = inputColor;
    break;
  case 'empty':
    var fillTexture = '#fff';
    break;
  case 'striped':
    var patternPath = (
      'M 5 -10 ' +
      'L 5  10 '
    );
    var fillTexture = paper.path(patternPath).attr({
      fill: 'none',
      stroke: inputColor,
      strokeWidth: 1,
    }).toPattern(0, 0, 8, 10);
    break;
}


// Draw each shape.
var shapes = [];
for (var i=0; i<inputNumberOfShapes; i++) {
  switch (inputShape) {
    case 'oval':
      shapes.push(drawOval(paper, cardCenter, shapeWidth, shapeHeight));
      break;
    case 'diamond':
      shapes.push(drawDiamond(paper, cardCenter, shapeWidth, shapeHeight));
      break;
    case 'bean':
      shapes.push(drawBean(paper, cardCenter, shapeWidth, shapeHeight));
      break;
  }
}


// Style each shape.
for (var i in shapes) {
  shapes[i].attr({
    fill: fillTexture,
    stroke: inputColor,
    strokeWidth: 6,
  });
}


// Move each shape.


// make a second bean and move it in place
/*
var secondBean = bean.clone();
secondBeanMatrix = new Snap.Matrix();
secondBeanMatrix.scale(beanScaleFactor);
secondBeanMatrix.translate(0, 360);
secondBean.transform(secondBeanMatrix);
*/

// Set the horizontal offset
/*
switch (inputNumberOfShapes) {
  case 1:
    var beanVerticalCenteringOffset = shapeHeight;
  case 2:
    var beanVerticalCenteringOffset = shapeHeight;
  case 3:
    var beanVerticalCenteringOffset = shapeHeight;
}
*/
