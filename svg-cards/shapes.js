function drawOval(paper, cardCenter, shapeWidth, shapeHeight) {
  // Draws an oval centered on the card.

  var ovalTopLeftCorner = [cardCenter[0] - shapeWidth / 2, cardCenter[1] - shapeHeight / 2],
      roundedness = shapeHeight / 2,
      oval = paper.rect(ovalTopLeftCorner[0], ovalTopLeftCorner[1], shapeWidth, shapeHeight, roundedness);

  return oval
}


function drawDiamond(paper, cardCenter, shapeWidth, shapeHeight) {
  // Draws a diamond centered on the card.
  var leftDiamondCorner = [cardCenter[0] - shapeWidth / 2, cardCenter[1]],
      diamond = paper.path(
         'M ' + leftDiamondCorner[0] + ' ' + leftDiamondCorner[1] + ' ' +
         'l 120 -55 ' +
         'l 120 55 ' +
         'l -120 55 ' +
         'l -120 -55 ' +
         'z'
       );

  return diamond
}


function drawBean(paper, cardCenter, shapeWidth, shapeHeight, beanScaleFactor) {
  // Draws a bean centered on the card

  // Set the horizontal offset so we can put the bean in the middle of the card.
  var beanHorizontalCenteringOffset = -1 * shapeWidth / 2 * beanScaleFactor;

  // This path starts at the leftmost point,
  // so only the x coordinate needs to be shifted in the initial move.
  var bean = paper.path(
    'M ' + (cardCenter[0] + beanHorizontalCenteringOffset) + ' ' + (cardCenter[1] + shapeHeight) + ' ' +
    'c -7   36,  0   55,  8   68 ' +
    'c  12  19,  33  18,  56  0 ' +
    'c  31 -24,  72 -18,  85 -10 ' +
    'c  34  20,  83  16,  98  12 ' +
    'c  34 -9,   55 -37,  62 -50 ' +
    'c  14 -26   13 -67,  1  -79 ' +
    'c -23 -23  -51  13, -73  18 ' +
    'c -20  4,  -38  9,  -61  0 ' +
    'c -24 -9,  -68 -24, -83 -25 ' +
    'c -33 -2,  -53  12, -65  21 ' +
    'c -14  10, -20  25, -27  43 ' +
    'z'
  );
  var beanMatrix = new Snap.Matrix();
  beanMatrix.scale(beanScaleFactor);
  bean.transform(beanMatrix);

  return bean
}
