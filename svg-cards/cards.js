function drawCard(inputNumberOfShapes, inputColor, inputTexture, inputShape) {
  // Draw a card.

  // Setup the drawing environment.
  var width = 800,
      height = 600,
      paper = Snap(width, height);


  // Setup the SVG background.
  var background = paper.rect(0, 0, width, height).attr({
    fillOpacity: 0.0,
  });


  // Make the card background.
  var cardAspectRatio = 0.64,
      cardHeight = 550,
      cardWidth = cardAspectRatio * cardHeight,  // 352
      cardCenter = [cardWidth / 2, cardHeight / 2],
      cardRoundedness = cardHeight / 28;  // 20

  var cardBackground = paper.rect(0, 0, cardWidth, cardHeight, cardRoundedness).attr({
    fill: '#fff',
    stroke: '#000',
  });


  // Shape parameters.
  var cardWidthToShapeWidthRatio = 1.477,
      shapeWidth = cardWidth / cardWidthToShapeWidthRatio,
      shapeAspectRatio = 2.133,
      shapeHeight = shapeWidth / shapeAspectRatio,
      // This scaling factor was determined by first building the bean path
      // and then measuring things to make certain length ratios correct.
      beanScaleFactor = 0.701;


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
        shapes.push(drawBean(paper, cardCenter, shapeWidth, shapeHeight, beanScaleFactor));
        break;
    }
  }


  // Style each shape.

  // Setup texture for use in striped fills.
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

  for (var i in shapes) {
    shapes[i].attr({
      fill: fillTexture,
      stroke: inputColor,
      strokeWidth: 6,
    });
  }


  // Move each shape.

  // Set the locations of the 2- and 3-shape vertical offsets.
  // The 1-shape offset is just zero, as is the shape in the middle of the 3-shape group.
  // Beans have their offsets tweaked a bit.
  var cardHeightToTwoShapeOffsetRatio = 3.524,
      cardHeightToThreeShapeOffsetRatio = 3.488;

  switch (inputNumberOfShapes) {
    case 2:
      var offset = cardHeight / cardHeightToTwoShapeOffsetRatio / 2,
          additionalBeanOffset = shapeHeight / 4;
      break;
    case 3:
      var offset = cardHeight / cardHeightToThreeShapeOffsetRatio,
          additionalBeanOffset = shapeHeight / 3;
      break;
  }

  // Create the transform matrices.
  // If we're drawing multiple beans, we have to apply the scaling again.
  var firstMatrix = new Snap.Matrix();
  var secondMatrix = new Snap.Matrix();
  if (inputShape == 'bean') {
    firstMatrix.scale(beanScaleFactor);
    secondMatrix.scale(beanScaleFactor);
    offset += additionalBeanOffset
  }

  if (inputNumberOfShapes > 1) {
    firstMatrix.translate(0, offset);
    shapes[0].transform(firstMatrix);
    secondMatrix.translate(0, -offset);
    shapes[1].transform(secondMatrix);
  }

}
