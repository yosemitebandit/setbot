var width = 800;
var height = 600;
var paper = Raphael('container', width, height);

var background = paper.rect(0, 0, width, height);
background.attr('fill-opacity', 0.0);

var aspectRatio = 0.625;
var cardHeight = 550;
var cardWidth = aspectRatio * cardHeight;
var cardRoundedness = 20;
var cardBackground = paper.rect(250, 25, cardWidth, cardHeight, cardRoundedness);
cardBackground.attr('fill', '#fff');
cardBackground.attr('stroke', '#000');

/*
// one green empty oval
var ovalWidth = 250;
var ovalHeight = 100;
var ovalRoundedness = ovalHeight / 2;
var oval = paper.rect(300, 250, ovalWidth, ovalHeight, ovalRoundedness);
oval.attr('fill', '#fff');
oval.attr('stroke', '#35bd2d');
oval.attr('stroke-width', '6');
*/

/*
// one red solid bean
var topHalfBean = paper.path(
  'M 280 300 ' +
  'c 0 0, 10 -60, 50 -60 ' +
  'c 40 0, 85 20, 125 20 ' +
  'c 95 0, 140 -70, 110 40 ' +
  'z'
);
var beanScaleFactor = 0.8;
topHalfBean.scale(beanScaleFactor, beanScaleFactor);
var bottomHalfBean = topHalfBean.clone();
bottomHalfBean.rotate(180);
bottomHalfBean.translate(9, -60);
var beanAttributes = {
  type: 'path',
  stroke: 'red',
  'stroke-width': '3',
  fill: 'red',
};
topHalfBean.attr(beanAttributes);
bottomHalfBean.attr(beanAttributes);
*/

// two purple solid diamonds
var topHalfDiamond = paper.path(
  'M 175 300 ' +
  'l 120 -50 ' +
  'l 120 50 ' +
  'z'
);
var bottomHalfDiamond = topHalfDiamond.clone();
var diamond = paper.set();
diamond.push(topHalfDiamond, bottomHalfDiamond);
diamond.transform('T 130 70');
bottomHalfDiamond.rotate(180);
bottomHalfDiamond.translate(0, -50);
diamond.attr({
  stroke: 'purple',
  fill: 'purple',
});
var secondDiamond = diamond.clone();
secondDiamond.transform('T 130 -70');
secondDiamond[1].rotate(180);
secondDiamond[1].translate(0, -50);


// save on "s"
$(function() {
  $('html').keypress(function(event) {
    console.log(event.which);
    if (event.which === 115) {
      var svgString = document.getElementById('container').innerHTML;
      a = document.createElement('a');
      a.download = 'card.svg';
      a.type = 'image/svg+xml';
      blob = new Blob([svgString], {"type": "image/svg+xml"});
      a.href = (window.URL || webkitURL).createObjectURL(blob);
      a.click();
    }
  });
});
